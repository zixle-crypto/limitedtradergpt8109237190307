from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="Zixle Studios", version="5.0.0")

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
  <title>Zixle Studios | Roblox Games</title>
  <meta name="description" content="Zixle Studios creates original Roblox games, player worlds, bacon characters, access passes, and merch drops." />
  <style>
    :root{color-scheme:dark;--bg:#070707;--panel:#101010;--ink:#fff8ef;--muted:#b8afa5;--red:#ff2b18;--gold:#ffd166;--ice:#71e9ff;--line:rgba(255,255,255,.16)}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}a{color:inherit;text-decoration:none}body:before{content:"";position:fixed;inset:0;z-index:-2;background:radial-gradient(circle at 50% 0,rgba(255,43,24,.28),transparent 33rem),radial-gradient(circle at 82% 18%,rgba(113,233,255,.18),transparent 25rem),linear-gradient(180deg,rgba(0,0,0,.2),#070707 74%)}body:after{content:"";position:fixed;inset:0;z-index:-1;opacity:.34;background-image:linear-gradient(rgba(255,255,255,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.055) 1px,transparent 1px);background-size:62px 62px;mask-image:linear-gradient(to bottom,#000,transparent 78%)}header{position:sticky;top:0;z-index:20;background:rgba(7,7,7,.78);backdrop-filter:blur(18px);border-bottom:1px solid var(--line)}nav{width:min(1180px,calc(100% - 34px));height:82px;margin:auto;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:24px}.nav-left,.nav-right{display:flex;gap:24px;align-items:center;color:var(--muted);font-weight:850}.nav-right{justify-content:flex-end}.logo{display:flex;align-items:center;gap:11px;font-size:1.08rem;font-weight:1000;text-transform:uppercase;letter-spacing:.04em}.logo-mark{width:48px;height:48px;border-radius:8px;display:grid;place-items:center;background:linear-gradient(135deg,var(--red),var(--gold));color:#160200;box-shadow:0 18px 50px rgba(255,43,24,.28);transform:rotate(-4deg)}.button{min-height:44px;display:inline-flex;align-items:center;justify-content:center;padding:0 16px;border-radius:8px;border:1px solid var(--line);font-weight:950;background:rgba(255,255,255,.06)}.button.hot{border:0;color:#1b0502;background:linear-gradient(135deg,var(--red),var(--gold))}.hero{width:min(1180px,calc(100% - 34px));min-height:calc(100vh - 82px);margin:auto;display:grid;place-items:center;text-align:center;padding:58px 0 36px}.eyebrow{display:inline-flex;align-items:center;gap:8px;margin-bottom:18px;padding:9px 12px;border:1px solid rgba(255,43,24,.38);background:rgba(255,43,24,.13);border-radius:8px;color:#ffd7d1;text-transform:uppercase;letter-spacing:.09em;font-weight:1000;font-size:.78rem}h1{margin:0 auto;max-width:980px;font-size:clamp(4rem,10vw,10.5rem);line-height:.82;letter-spacing:0;text-transform:uppercase;text-shadow:0 24px 80px rgba(0,0,0,.68)}.hero h1 span{display:block;color:var(--red)}.lead{max-width:720px;margin:26px auto 0;color:#ded4c9;font-size:1.18rem;line-height:1.7}.hero-actions{display:flex;justify-content:center;flex-wrap:wrap;gap:13px;margin-top:30px}.stage{width:100%;margin-top:48px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:linear-gradient(135deg,rgba(255,43,24,.22),rgba(113,233,255,.1)),#111;box-shadow:0 32px 90px rgba(0,0,0,.55)}.stage-art{min-height:300px;display:grid;grid-template-columns:1fr 1.2fr 1fr;align-items:end;gap:14px;padding:28px;background:linear-gradient(180deg,rgba(255,255,255,.08),transparent),repeating-linear-gradient(90deg,rgba(255,255,255,.08) 0 2px,transparent 2px 42px)}.roblox{position:relative;height:218px;filter:drop-shadow(0 22px 22px rgba(0,0,0,.45))}.roblox:before{content:"";position:absolute;left:50%;top:0;width:72px;height:72px;transform:translateX(-50%);border-radius:8px;background:#f3f3f3;border:4px solid #0a0a0a}.roblox:after{content:"";position:absolute;left:50%;top:62px;width:96px;height:86px;transform:translateX(-50%);border-radius:8px;background:linear-gradient(135deg,#111,#2b2b2b);border:4px solid #0a0a0a;box-shadow:-58px 20px 0 -18px #f3f3f3,58px 20px 0 -18px #f3f3f3,-28px 106px 0 -10px #14333c,28px 106px 0 -10px #14333c}.roblox.left{transform:rotate(-8deg)}.roblox.right{transform:rotate(7deg)}.snowball{align-self:center;justify-self:center;width:min(32vw,260px);aspect-ratio:1;border-radius:50%;background:radial-gradient(circle at 35% 30%,#fff,#c9f5ff 42%,#65dbff 76%);box-shadow:0 0 0 12px rgba(255,255,255,.1),0 0 90px rgba(113,233,255,.55)}.caption{display:flex;justify-content:space-between;gap:16px;padding:18px 20px;border-top:1px solid var(--line);background:rgba(0,0,0,.35);font-weight:900}.section{width:min(1180px,calc(100% - 34px));margin:auto;padding:86px 0}.split{display:grid;grid-template-columns:.85fr 1.15fr;gap:44px;align-items:start}.kicker{margin:0 0 12px;color:var(--red);font-weight:1000;text-transform:uppercase;letter-spacing:.08em}.section h2{margin:0;font-size:clamp(2.5rem,5.4vw,6.2rem);line-height:.88;text-transform:uppercase}.copy{color:var(--muted);font-size:1.06rem;line-height:1.78}.copy p{margin:0 0 16px}.stats{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.stat{padding:34px 20px;text-align:center;border-right:1px solid var(--line);background:rgba(255,255,255,.035)}.stat:last-child{border-right:0}.stat strong{display:block;font-size:clamp(2rem,4vw,4.2rem);line-height:1;color:#fff}.stat span{display:block;margin-top:8px;color:var(--muted);font-weight:850}.games-head{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:24px}.games{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.game{min-height:360px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#101010;display:flex;flex-direction:column}.thumb{min-height:190px;background:linear-gradient(135deg,var(--red),var(--gold));position:relative;overflow:hidden}.thumb:before{content:"";position:absolute;inset:22px;border-radius:10px;background:radial-gradient(circle at 36% 32%,#fff,#dff8ff 32%,#8feaff 58%,transparent 60%)}.thumb:after{content:"";position:absolute;left:24px;right:24px;bottom:22px;height:46px;background:repeating-linear-gradient(90deg,#fff 0 26px,#0a0a0a 26px 34px);border-radius:8px;opacity:.82}.thumb.dark{background:linear-gradient(135deg,#0c0c0c,#777)}.thumb.ice{background:linear-gradient(135deg,#13c8ff,#fff)}.thumb.fire{background:linear-gradient(135deg,#071022,#ff3b18 62%,#ffd166)}.game-body{padding:18px}.game h3{margin:0 0 8px;font-size:1.25rem}.game p{margin:0;color:var(--muted);line-height:1.5}.badge{display:inline-flex;margin-top:auto;margin-left:18px;margin-bottom:18px;padding:7px 9px;border-radius:8px;background:rgba(255,255,255,.08);border:1px solid var(--line);font-size:.78rem;font-weight:950;color:#fff}.contact{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:linear-gradient(135deg,rgba(255,43,24,.16),rgba(255,255,255,.05))}.contact-inner{display:grid;grid-template-columns:1fr 1fr;gap:0}.contact-copy{padding:38px}.form{display:grid;gap:12px;padding:38px;background:rgba(0,0,0,.28);border-left:1px solid var(--line)}input,textarea{width:100%;border:1px solid var(--line);border-radius:8px;background:rgba(255,255,255,.07);color:var(--ink);font:inherit;padding:14px}textarea{min-height:132px;resize:vertical}footer{border-top:1px solid var(--line);padding:36px 0;color:var(--muted)}footer .inner{width:min(1180px,calc(100% - 34px));margin:auto;display:flex;justify-content:space-between;gap:20px}.source-note{position:absolute;left:-9999px}@media(max-width:920px){nav{grid-template-columns:1fr auto}.nav-left{display:none}.split,.contact-inner{grid-template-columns:1fr}.games,.stats{grid-template-columns:repeat(2,1fr)}.form{border-left:0;border-top:1px solid var(--line)}.stage-art{grid-template-columns:1fr}.roblox.right{display:none}.snowball{width:210px}}@media(max-width:560px){nav{height:70px}.nav-right a:not(.button){display:none}.hero{min-height:auto;padding-top:64px}h1{font-size:clamp(3.3rem,18vw,5.5rem)}.stats,.games{grid-template-columns:1fr}.stat{border-right:0;border-bottom:1px solid var(--line)}.caption,footer .inner{display:block}.section{padding:62px 0}.stage-art{min-height:260px}}
  </style>
</head>
<body>
  <header>
    <nav>
      <div class="nav-left"><a href="#about">About Us</a><a href="#games">Games</a></div>
      <a class="logo" href="/"><span class="logo-mark">ZX</span><span>Zixle Studios</span></a>
      <div class="nav-right"><a href="#shop">Marketplace</a><a href="#contact">Contact</a><a class="button hot" href="#games">View Games</a></div>
    </nav>
  </header>
  <main>
    <section class="hero">
      <div>
        <span class="eyebrow">Roblox game studio</span>
        <h1>We Make <span>Zixle Games.</span></h1>
        <p class="lead">Original Roblox worlds with bacon characters, wild thumbnails, player access, merch drops, game passes, and updates built for players who come back.</p>
        <div class="hero-actions"><a class="button hot" href="#games">View Games</a><a class="button" href="#about">About Us</a></div>
        <div class="stage" aria-label="Roblox style game preview">
          <div class="stage-art"><div class="roblox left"></div><div class="snowball"></div><div class="roblox right"></div></div>
          <div class="caption"><span>Snowball chaos</span><span>Bacon character energy</span><span>Roblox-first worlds</span></div>
        </div>
      </div>
    </section>
    <section class="section" id="about">
      <div class="split">
        <div><p class="kicker">About Us</p><h2>Player worlds with a loud identity.</h2></div>
        <div class="copy"><p>Zixle Studios creates its own Roblox games. The site is built like a player hub, not a generic company page: bold game art, clear launches, featured modes, access passes, and community-ready merch.</p><p>Every game should feel easy to understand from the first second: who the characters are, what the conflict is, what players can unlock, and why they should jump in with friends.</p><p><a class="button hot" href="#contact">Contact Zixle</a></p></div>
      </div>
    </section>
    <section class="stats" aria-label="Zixle player stats">
      <div class="stat"><strong>24/7</strong><span>World Access</span></div>
      <div class="stat"><strong>4+</strong><span>Game Concepts</span></div>
      <div class="stat"><strong>100%</strong><span>Roblox Focus</span></div>
      <div class="stat"><strong>VIP</strong><span>Passes + Drops</span></div>
    </section>
    <section class="section" id="games">
      <div class="games-head"><div><p class="kicker">Featured Games</p><h2>Check out our Zixle worlds.</h2></div><a class="button" href="#contact">Suggest a mode</a></div>
      <div class="games">
        <article class="game"><div class="thumb ice"></div><div class="game-body"><h3>Snowball Chase</h3><p>Bacon players sprint across icy maps while rivals line up the perfect hit.</p></div><span class="badge">In development</span></article>
        <article class="game"><div class="thumb dark"></div><div class="game-body"><h3>U Got Smoked</h3><p>A dramatic battle mode made for quick rounds, loud wins, and thumbnail moments.</p></div><span class="badge">Featured</span></article>
        <article class="game"><div class="thumb fire"></div><div class="game-body"><h3>Fireball Arena</h3><p>Ice versus fire powers with destructible-feeling arenas and high-impact rounds.</p></div><span class="badge">Concept</span></article>
        <article class="game"><div class="thumb"></div><div class="game-body"><h3>Bacon City</h3><p>A social Roblox world for cosmetics, hangouts, codes, shops, and player events.</p></div><span class="badge">Coming soon</span></article>
      </div>
    </section>
    <section class="section" id="shop">
      <div class="split">
        <div><p class="kicker">Marketplace</p><h2>Access, merch, and drops.</h2></div>
        <div class="copy"><p>Players should see what they can unlock: VIP doors, early access areas, cosmetic packs, creator merch, limited event rewards, and bacon starter bundles.</p><p>The marketplace section is designed to feel connected to Roblox gameplay instead of just being a normal shop.</p></div>
      </div>
    </section>
    <section class="section" id="contact">
      <div class="contact"><div class="contact-inner"><div class="contact-copy"><p class="kicker">Build With Zixle</p><h2>Got a world idea?</h2><p class="copy">Send Zixle a note about game modes, merch ideas, access passes, or Roblox community features players would want next.</p></div><form class="form"><input placeholder="Name" aria-label="Name"><input placeholder="Email" aria-label="Email"><input placeholder="Subject" aria-label="Subject"><textarea placeholder="Message" aria-label="Message"></textarea><button class="button hot" type="button">Send Message</button></form></div></div>
    </section>
  </main>
  <footer><div class="inner"><strong>Zixle Studios</strong><span>Original Roblox games, player access, bacon characters, and marketplace drops.</span></div></footer>
  <span class="source-note">Reference structure inspired by basedgames.com: hero, about, stats, featured games, and contact flow.</span>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HOME_HTML)

@app.get("/health")
def health():
    return {"ok": True, "studio": "Zixle Studios", "type": "original Roblox games", "theme": "based-style studio landing"}
