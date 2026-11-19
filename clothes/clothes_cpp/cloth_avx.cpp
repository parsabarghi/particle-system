#include <SDL2/SDL.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

// ---------- Data ----------
struct Particle {
    float x = 0, y = 0;
    float prevx = 0, prevy = 0;
    bool  isPinned = false;
};

struct Stick {
    int   p1, p2;   // indices into particles[]
    float length;   // rest length
};

// ---------- Segment intersection (verbatim from Python) ----------
static int orientation(float px, float py, float qx, float qy, float rx, float ry) {
    float val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy);
    if (val == 0.0f) return 0;
    return (val > 0.0f) ? 1 : 2;
}
static bool onSegment(float px, float py, float qx, float qy, float rx, float ry) {
    return qx <= std::max(px, rx) && qx >= std::min(px, rx)
        && qy <= std::max(py, ry) && qy >= std::min(py, ry);
}
static bool intersect(float x1, float y1, float x2, float y2,
                      float x3, float y3, float x4, float y4) {
    int o1 = orientation(x1, y1, x2, y2, x3, y3);
    int o2 = orientation(x1, y1, x2, y2, x4, y4);
    int o3 = orientation(x3, y3, x4, y4, x1, y1);
    int o4 = orientation(x3, y3, x4, y4, x2, y2);
    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && onSegment(x1, y1, x3, y3, x2, y2)) return true;
    if (o2 == 0 && onSegment(x1, y1, x4, y4, x2, y2)) return true;
    if (o3 == 0 && onSegment(x3, y3, x1, y1, x4, y4)) return true;
    if (o4 == 0 && onSegment(x3, y3, x2, y2, x4, y4)) return true;
    return false;
}

// ---------- Cloth ----------
class Cloth {
public:
    std::vector<Particle> particles;
    std::vector<Stick>    sticks;

    int   clothW, clothH;
    float spacing;
    float screenW, screenH;
    float gravityY = 981.0f;
    float drag     = 0.01f;
    Uint32 lastUpdateTime = 0;
    float  deltaTime      = 0.01f;

    Cloth(float w, float h, float sp, float sh, float sw)
        : clothW((int)(w / sp)), clothH((int)(h / sp)),
          spacing(sp), screenW(sw), screenH(sh) {}

    void setup() {
        particles.reserve(clothW * clothH);
        for (int i = 0; i < clothW; ++i) {
            for (int j = 0; j < clothH; ++j) {
                Particle p;
                p.x = p.prevx = i * spacing;
                p.y = p.prevy = j * spacing;
                particles.push_back(p);

                if (i > 0) {
                    int a = (i - 1) * clothH + j;
                    int b =  i      * clothH + j;
                    float dx = particles[b].x - particles[a].x;
                    float dy = particles[b].y - particles[a].y;
                    sticks.push_back({a, b, std::sqrt(dx*dx + dy*dy)});
                }
                if (j > 0) {
                    int a = i * clothH + (j - 1);
                    int b = i * clothH +  j;
                    float dx = particles[b].x - particles[a].x;
                    float dy = particles[b].y - particles[a].y;
                    sticks.push_back({a, b, std::sqrt(dx*dx + dy*dy)});
                }
                if (j == 0 && i % 2 == 0) {
                    particles.back().isPinned = true;
                }
            }
        }
        lastUpdateTime = SDL_GetTicks();
    }

    void update(float mx, float my, float pmx, float pmy) {
        Uint32 now = SDL_GetTicks();
        deltaTime = (now - lastUpdateTime) / 1000.0f;
        lastUpdateTime = now;

        // Verlet integration
        for (auto& p : particles) {
            if (p.isPinned) {
                p.x = p.prevx;
                p.y = p.prevy;
                continue;
            }
            float oldx = p.x, oldy = p.y;
            p.x += (p.x - p.prevx) * (1.0f - drag);
            p.y += (p.y - p.prevy) * (1.0f - drag) + gravityY * deltaTime * deltaTime;
            p.prevx = oldx;
            p.prevy = oldy;

            if (p.y >= screenH) p.y = screenH;
            if (p.x >= screenW) p.x = screenW;
            if (p.y < 0)        p.y = 0;
            if (p.x < 0)        p.x = 0;
        }

        // Constraint solving + mouse cutting (5 iterations)
        std::vector<char> cut(sticks.size(), 0);
        for (int iter = 0; iter < 5; ++iter) {
            for (size_t k = 0; k < sticks.size(); ++k) {
                if (cut[k]) continue;
                Stick& s = sticks[k];
                Particle& a = particles[s.p1];
                Particle& b = particles[s.p2];

                float dx = a.x - b.x;
                float dy = a.y - b.y;
                float len = std::sqrt(dx*dx + dy*dy);
                if (len < 1e-6f) continue;

                float factor = (s.length - len) / len * 0.5f;
                float ox = dx * factor;
                float oy = dy * factor;

                a.x += ox; a.y += oy;
                b.x -= ox; b.y -= oy;

                if (intersect(mx, my, pmx, pmy, a.x, a.y, b.x, b.y)) {
                    cut[k] = 1;
                }
            }
        }
        // Compact sticks, dropping cut ones
        size_t w = 0;
        for (size_t k = 0; k < sticks.size(); ++k)
            if (!cut[k]) sticks[w++] = sticks[k];
        sticks.resize(w);
    }

    void render(SDL_Renderer* ren) const {
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        for (const auto& s : sticks) {
            const Particle& a = particles[s.p1];
            const Particle& b = particles[s.p2];
            SDL_RenderDrawLine(ren, (int)a.x, (int)a.y, (int)b.x, (int)b.y);
        }
        for (const auto& p : particles) {
            fillCircle(ren, (int)p.x, (int)p.y, 5);
        }
    }

private:
    static void fillCircle(SDL_Renderer* ren, int cx, int cy, int r) {
        for (int y = -r; y <= r; ++y) {
            int dx = (int)std::sqrt((float)(r*r - y*y));
            SDL_RenderDrawLine(ren, cx - dx, cy + y, cx + dx, cy + y);
        }
    }
};

// ---------- main ----------
int main() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::printf("SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    const int SW = 800, SH = 600;
    SDL_Window* win = SDL_CreateWindow("Cloth Sim",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        SW, SH, SDL_WINDOW_SHOWN);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    if (!win || !ren) { SDL_Quit(); return 1; }

    Cloth cloth(800, 600, 25, SH, SW);
    cloth.setup();

    bool  running   = true;
    bool  mouseDown = false;
    float mx = 0, my = 0;
    float pmx = 0, pmy = 0;

    Uint32 titleTimer = SDL_GetTicks();
    int    frames = 0;

    while (running) {
        Uint32 frameStart = SDL_GetTicks();

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                mouseDown = true;
                pmx = mx; pmy = my;
            }
            else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
                mouseDown = false;
            }
        }

        if (mouseDown) {
            int imx, imy;
            SDL_GetMouseState(&imx, &imy);
            mx = (float)imx; my = (float)imy;
        }

        cloth.update(mx, my, pmx, pmy);

        SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
        SDL_RenderClear(ren);
        cloth.render(ren);
        SDL_RenderPresent(ren);

        // FPS in window title
        ++frames;
        Uint32 now = SDL_GetTicks();
        if (now - titleTimer >= 500) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "Cloth Sim - FPS: %.1f",
                          frames * 1000.0f / (now - titleTimer));
            SDL_SetWindowTitle(win, buf);
            frames = 0;
            titleTimer = now;
        }

        // ~60 FPS cap
        Uint32 elapsed = SDL_GetTicks() - frameStart;
        if (elapsed < 16) SDL_Delay(16 - elapsed);
    }

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}