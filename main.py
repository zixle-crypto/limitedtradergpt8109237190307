from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="Zixle Studios", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Zixle Studios | Roblox Game Development</title>
  <meta name="description" content="Zixle Studios builds Roblox games, access systems, marketplace merch flows, game UI, economy tools, and launch-ready prototypes." />
  <style>
    :root {
      color-scheme: dark;
      --bg: #07100f;
      --panel: rgba(15, 31, 29, 0.78);
      --panel-strong: rgba(18, 37, 35, 0.94);
      --text: #f4fffb;
      --muted: #a7bdb8;
      --line: rgba(145, 226, 209, 0.2);
      --teal: #5ce6ca;
      --amber: #ffbd62;
      --blue: #72a7ff;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 18% 12%, rgba(92, 230, 202, 0.2), transparent 34rem),
        radial-gradient(circle at 88% 12%, rgba(255, 189, 98, 0.14), transparent 32rem),
        linear-gradient(180deg, #081311 0%, #07100f 48%, #0a1215 100%);
      color: var(--text);
    }
    a { color: inherit; text-decoration: none; }
    .grid {
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.34;
      background-image: linear-gradient(rgba(92,230,202,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(92,230,202,.08) 1px, transparent 1px);
      background-size: 72px 72px;
      mask-image: linear-gradient(to bottom, black 0%, transparent 78%);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      backdrop-filter: blur(18px);
      background: rgba(7, 16, 15, 0.82);
      border-bottom: 1px solid var(--line);
    }
    nav {
      width: min(1180px, calc(100% - 40px));
      height: 76px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 22px;
    }
    .brand { display: inline-flex; align-items: center; gap: 12px; font-weight: 900; }
    .mark {
      width: 38px;
      height: 38px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(92,230,202,.38);
      border-radius: 8px;
      color: var(--teal);
      background: linear-gradient(145deg, rgba(92,230,202,.22), rgba(255,189,98,.12));
      box-shadow: 0 12px 38px rgba(92,230,202,.12);
    }
    .links { display: flex; align-items: center; gap: 24px; color: var(--muted); font-weight: 750; }
    .links a:hover { color: var(--text); }
    .cta {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 0 18px;
      border-radius: 8px;
      border: 1px solid rgba(92,230,202,.36);
      background: rgba(92,230,202,.14);
      color: #e7fff8;
      font-weight: 900;
    }
    main { position: relative; }
    .hero {
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
      min-height: calc(100vh - 76px);
      display: grid;
      grid-template-columns: minmax(0, .96fr) minmax(420px, 1.04fr);
      align-items: center;
      gap: 44px;
      padding: 54px 0 50px;
    }
    h1 {
      margin: 0;
      max-width: 760px;
      font-size: clamp(2.8rem, 5.25vw, 4.55rem);
      line-height: .96;
      letter-spacing: 0;
      text-wrap: balance;
    }
    .lead {
      max-width: 625px;
      margin: 28px 0 0;
      color: #bfdbd5;
      font-size: 1.18rem;
      line-height: 1.68;
    }
    .actions { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 34px; }
    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 50px;
      padding: 0 22px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.05);
      font-weight: 900;
    }
    .button.primary { color: #06231e; border: 0; background: linear-gradient(135deg, var(--teal), #a0f7df); box-shadow: 0 16px 44px rgba(92,230,202,.22); }
    .board {
      min-height: 480px;
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(92,230,202,.1), transparent 38%), linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
      box-shadow: 0 26px 90px rgba(0,0,0,.38);
    }
    .board:before {
      content: "";
      position: absolute;
      inset: 0;
      background-image: linear-gradient(rgba(244,255,251,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(244,255,251,.06) 1px, transparent 1px);
      background-size: 44px 44px;
      transform: perspective(700px) rotateX(58deg) translateY(58px) scale(1.25);
      transform-origin: bottom;
    }
    .node {
      position: absolute;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-strong);
      padding: 14px;
      box-shadow: 0 18px 54px rgba(0,0,0,.32);
    }
    .node h2, .node h3 { margin: 0; font-size: .95rem; }
    .node p { margin: 8px 0 0; color: var(--muted); font-size: .8rem; line-height: 1.48; }
    .node.one { left: 42px; top: 42px; width: 270px; }
    .node.two { right: 34px; top: 94px; width: 232px; }
    .node.three { left: 64px; bottom: 42px; width: 305px; }
    .node.four { right: 54px; bottom: 72px; width: 220px; }
    .play {
      position: absolute;
      right: 30%;
      top: 34%;
      width: 152px;
      aspect-ratio: 1;
      display: grid;
      place-items: center;
      border: 1px solid rgba(255,189,98,.42);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(255,189,98,.2), transparent), rgba(10,17,20,.72);
      color: var(--amber);
      font-weight: 950;
    }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px; }
    .metric { padding: 10px; border-radius: 8px; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08); color: var(--muted); font-size: .76rem; }
    .metric strong { display: block; color: var(--text); font-size: 1rem; margin-bottom: 3px; }
    .band { border-top: 1px solid var(--line); background: rgba(255,255,255,.025); }
    .section { width: min(1180px, calc(100% - 40px)); margin: 0 auto; padding: 78px 0; }
    .section-head { display: flex; justify-content: space-between; align-items: end; gap: 24px; margin-bottom: 30px; }
    h2 { margin: 0; max-width: 730px; font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1; }
    .section-head p { max-width: 440px; margin: 0; color: var(--muted); line-height: 1.65; }
    .services { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
    .service { min-height: 198px; padding: 20px; border-radius: 8px; border: 1px solid var(--line); background: rgba(255,255,255,.04); }
    .service span { display: inline-grid; place-items: center; width: 38px; height: 38px; margin-bottom: 28px; border-radius: 8px; background: rgba(92,230,202,.12); color: var(--teal); font-weight: 950; }
    .service h3 { margin: 0 0 10px; font-size: 1rem; }
    .service p { margin: 0; color: var(--muted); line-height: 1.55; font-size: .93rem; }
    .work { display: grid; grid-template-columns: .94fr 1.06fr; gap: 18px; }
    .case, .api { border-radius: 8px; border: 1px solid var(--line); background: var(--panel); padding: 28px; min-height: 280px; }
    .case h3 { margin: 0 0 12px; font-size: 1.55rem; }
    .case p { margin: 0; color: var(--muted); line-height: 1.65; }
    .step { display: flex; justify-content: space-between; gap: 12px; padding: 14px 0; margin-top: 10px; border-top: 1px solid var(--line); color: var(--muted); font-weight: 750; }
    .step strong { color: var(--text); }
    pre { margin: 0; white-space: pre-wrap; color: #cbefe6; line-height: 1.7; font-size: .94rem; }
    footer { width: min(1180px, calc(100% - 40px)); margin: 0 auto; padding: 34px 0 46px; color: var(--muted); display: flex; justify-content: space-between; gap: 20px; border-top: 1px solid var(--line); }
    @media (max-width: 1050px) { .services { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 940px) {
      .links { display: none; }
      .hero { grid-template-columns: 1fr; padding-top: 52px; }
      .board { min-height: 450px; }
      .work { grid-template-columns: 1fr; }
      .section-head { display: block; }
      .section-head p { margin-top: 16px; }
    }
    @media (max-width: 620px) {
      nav, .hero, .section, footer { width: min(100% - 28px, 1180px); }
      .cta { display: none; }
      h1 { font-size: clamp(2.65rem, 14vw, 4rem); }
      .lead { font-size: 1.04rem; }
      .actions { display: grid; }
      .button { width: 100%; }
      .services { grid-template-columns: 1fr; }
      .node { position: relative; inset: auto !important; width: auto !important; margin: 14px; }
      .board { min-height: auto; padding: 10px 0; }
      .board:before, .play { display: none; }
      footer { display: block; }
      footer span { display: block; margin-top: 10px; }
    }
  </style>
</head>
<body>
  <div class="grid"></div>
  <header>
    <nav aria-label="Main navigation">
      <a class="brand" href="/" aria-label="Zixle Studios home"><span class="mark">ZX</span><span>Zixle Studios</span></a>
      <div class="links"><a href="#work">Work</a><a href="#services">Services</a><a href="#marketplace">Marketplace</a><a href="#api">API</a><a href="mailto:hello@zixlestudios.com">Contact</a></div>
      <a class="cta" href="mailto:hello@zixlestudios.com">Start a build</a>
    </nav>
  </header>
  <main>
    <section class="hero" aria-labelledby="hero-title">
      <div>
        <h1 id="hero-title">Roblox games with access, style, and systems.</h1>
        <p class="lead">Zixle Studios builds Roblox experiences, game access flows, merch marketplace systems, UI, economy tools, and launch-ready prototypes for creators who want ideas to feel playable fast.</p>
        <div class="actions"><a class="button primary" href="mailto:hello@zixlestudios.com">Build with Zixle</a><a class="button" href="#api">View live tools</a></div>
      </div>
      <div class="board" aria-label="Roblox development systems board">
        <div class="play">PLAY</div>
        <article class="node one"><h2>Roblox Game Blueprint</h2><p>World access, progression, store loops, and rewards designed before the first sprint.</p><div class="metrics"><div class="metric"><strong>5</strong>systems</div><div class="metric"><strong>Pass</strong>access</div><div class="metric"><strong>Live</strong>API</div></div></article>
        <article class="node two"><h3>Gameplay Logic</h3><p>Controller states, access checks, inventory hooks, and server-safe routes.</p></article>
        <article class="node three"><h3>Marketplace Tools</h3><p>Merch drops, item lookup, inventory checks, and economy helpers stay online.</p></article>
        <article class="node four"><h3>Creator Launch Stack</h3><p>Prototype, access, merch, test, publish.</p></article>
      </div>
    </section>
    <section class="band" id="services"><div class="section"><div class="section-head"><h2>Development support for Roblox worlds.</h2><p>Focused help across the parts that make an experience feel real: game access, scripting, interfaces, marketplace systems, backend tools, and launch polish.</p></div><div class="services">
      <article class="service"><span>01</span><h3>Roblox Experiences</h3><p>World structure, gameplay loops, progression, and player-ready scripting.</p></article>
      <article class="service"><span>02</span><h3>Access Systems</h3><p>Game pass checks, role gates, private areas, rewards, and permissions.</p></article>
      <article class="service" id="marketplace"><span>03</span><h3>Marketplace & Merch</h3><p>Shop flows, merch drops, item pages, inventory hooks, and purchase logic.</p></article>
      <article class="service"><span>04</span><h3>Game UI</h3><p>Menus, HUDs, shops, progression screens, and readable in-game flows.</p></article>
      <article class="service"><span>05</span><h3>Backend Tools</h3><p>Fast APIs, data bridges, admin helpers, and analytics-ready endpoints.</p></article>
    </div></div></section>
    <section class="section" id="work"><div class="work"><article class="case"><h3>From rough Roblox idea to playable launch.</h3><p>Zixle Studios can help shape the flow, build the core mechanics, add access systems, connect marketplace merch, and clean up the parts players notice first.</p><div class="step"><strong>Plan</strong><span>core loop, access, and player goals</span></div><div class="step"><strong>Build</strong><span>systems, UI, merch, and API support</span></div><div class="step"><strong>Polish</strong><span>feedback, responsiveness, launch pass</span></div></article><article class="api" id="api"><pre>{
  "studio": "Zixle Studios",
  "focus": ["Roblox development", "game access", "marketplace merch"],
  "status": "/health",
  "live_endpoints": ["/market/item/search?q=dominus", "/market/item/{item_id}", "/roblox/player/{username}/inventory"]
}</pre></article></div></section>
  </main>
  <footer><strong>Zixle Studios</strong><span>Roblox game development, tools, access systems, and marketplace polish.</span></footer>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HOME_HTML)


@app.get("/health")
def health():
    return {"ok": True, "studio": "Zixle Studios"}


@app.get("/api")
def api_info():
    return {
        "studio": "Zixle Studios",
        "focus": ["Roblox development", "game access", "marketplace merch", "game UI"],
        "status": "online",
    }
