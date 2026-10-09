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

HOME_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Zixle Studios</title>
<meta name="description" content="Zixle Studios makes original Roblox games: Snowball Chase, U Got Smoked, Fireball Arena and Bacon City. Fly through our worlds.">
<meta name="theme-color" content="#5cc8ff">
<meta property="og:title" content="Zixle Studios | Roblox Games">
<meta property="og:description" content="Original Roblox worlds with bacon characters, game passes, merch drops and loud updates.">
<meta property="og:url" content="https://www.zixlestudios.com/">
<meta property="og:type" content="website">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect x='4' y='4' width='56' height='56' fill='%23ffd23a' stroke='%23142046' stroke-width='6'/%3E%3Cpath d='M18 18h28v8L28 40h18v8H18v-8l18-14H18z' fill='%23142046'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lilita+One&family=Nunito:wght@600;800;900&display=swap" rel="stylesheet">
<style>
/* Layout: a fixed real-time voxel world behind the page; each section is a stop the camera flies to, with chunky blocky game-UI panels on top. */
:root{
  color-scheme:light;
  --ink:#142046;        /* outlines, text */
  --sky:#5cc8ff;        /* upper sky */
  --haze:#c9f0ff;       /* horizon */
  --sun:#ffd23a;        /* primary buttons, logo */
  --grass:#4fd26b;
  --snow:#eaf8ff;
  --ice:#39c6ff;
  --fire:#ff6a1a;
  --pink:#ff4f9a;
  --smoke:#8a5cff;
  --paper:#ffffff;
  --muted:#4a5a85;
  --display:'Lilita One','Arial Black',Impact,system-ui,sans-serif;
  --body:'Nunito',system-ui,-apple-system,'Segoe UI',sans-serif;
  --edge:3px solid var(--ink);
  --lift:0 6px 0 var(--ink);
}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%;overflow-x:clip}
body{font-family:var(--body);font-weight:600;color:var(--ink);line-height:1.5;overflow-x:clip;
  background:linear-gradient(180deg,var(--sky) 0%,#8fdcff 45%,var(--haze) 100%) fixed;}
a{color:inherit}
img,canvas{max-width:100%}
:focus-visible{outline:4px solid var(--pink);outline-offset:3px}
[hidden]{display:none!important}
::selection{background:var(--sun);color:var(--ink)}

#world{position:fixed;inset:0;width:100%;height:100%;z-index:0;display:block;touch-action:pan-y}
.page{position:relative;z-index:2;pointer-events:none}
.page a,.page button,.page input,.page textarea,.page label,.page .panel{pointer-events:auto}
.wrap{width:min(1200px,100% - 32px);margin-inline:auto}

/* ---------- top bar ---------- */
.bar{position:fixed;z-index:10;top:0;left:0;right:0;padding:calc(env(safe-area-inset-top,0px) + 14px) 16px 0;pointer-events:none}
.bar-in{width:min(1200px,100%);margin-inline:auto;display:flex;align-items:center;justify-content:space-between;gap:12px}
.logo{pointer-events:auto;display:flex;align-items:center;gap:10px;text-decoration:none;background:var(--paper);border:var(--edge);box-shadow:var(--lift);padding:6px 14px 6px 6px}
.logo b{font-family:var(--display);font-weight:400;font-size:1.25rem;background:var(--sun);border:var(--edge);padding:0 8px;line-height:1.5}
.logo span{font-family:var(--display);font-size:1.15rem;letter-spacing:.01em}
.menu{pointer-events:auto;display:flex;gap:8px}
.menu a{font-family:var(--display);font-size:1rem;text-decoration:none;background:var(--paper);border:var(--edge);box-shadow:0 4px 0 var(--ink);padding:8px 14px;transition:transform .12s,box-shadow .12s}
.menu a:hover{transform:translateY(-2px);box-shadow:0 6px 0 var(--ink)}
.menu a:active{transform:translateY(4px);box-shadow:0 0 0 var(--ink)}
.menu a.play{background:var(--sun)}
.menu-btn{display:none;pointer-events:auto;font-family:var(--display);font-size:1rem;background:var(--paper);border:var(--edge);box-shadow:0 4px 0 var(--ink);padding:8px 14px;color:var(--ink)}
@media (max-width:820px){
  .menu-btn{display:block}
  .menu{position:fixed;left:16px;right:16px;top:calc(env(safe-area-inset-top,0px) + 76px);flex-direction:column;background:var(--paper);border:var(--edge);box-shadow:var(--lift);padding:10px;visibility:hidden;opacity:0;transform:translateY(-8px);transition:.18s}
  .menu.open{visibility:visible;opacity:1;transform:none}
  .menu a{box-shadow:none;text-align:center;font-size:1.15rem}
}

/* ---------- island tracker (where you are in the world) ---------- */
.tracker{position:fixed;z-index:9;right:16px;top:50%;transform:translateY(-50%);display:flex;flex-direction:column;gap:8px;pointer-events:auto}
.tracker a{display:block;width:18px;height:18px;border:var(--edge);background:var(--paper);box-shadow:0 3px 0 var(--ink);position:relative;transition:background .2s,transform .2s}
.tracker a.on{background:var(--sun);transform:scale(1.25)}
.tracker a span{position:absolute;right:28px;top:50%;transform:translateY(-50%);white-space:nowrap;font-family:var(--display);font-size:.9rem;background:var(--ink);color:var(--paper);padding:3px 8px;opacity:0;pointer-events:none;transition:opacity .15s}
.tracker a:hover span,.tracker a:focus-visible span{opacity:1}
@media (max-width:820px){.tracker{display:none}}

/* ---------- shared bits ---------- */
.stop{min-height:100vh;min-height:100svh;display:flex;align-items:center;padding-block:110px 60px}
.stop.right{justify-content:flex-end}
.panel{background:var(--paper);border:var(--edge);box-shadow:0 8px 0 var(--ink);padding:clamp(22px,3.4vw,36px);width:min(500px,100%);position:relative}
.big-type{font-family:var(--display);font-weight:400;line-height:.95;letter-spacing:.005em;text-wrap:balance}
h2.big-type{font-size:clamp(2.3rem,5.2vw,3.7rem);margin-bottom:14px}
.panel p{font-size:1.06rem;color:var(--muted);max-width:60ch}
.panel p + p{margin-top:10px}
.panel strong{color:var(--ink);font-weight:900}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;font-family:var(--display);font-weight:400;font-size:1.2rem;text-decoration:none;color:var(--ink);background:var(--sun);border:var(--edge);box-shadow:0 6px 0 var(--ink);padding:12px 22px;cursor:pointer;transition:transform .1s,box-shadow .1s,filter .1s;user-select:none}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 0 var(--ink);filter:brightness(1.05)}
.btn:active{transform:translateY(6px);box-shadow:0 0 0 var(--ink)}
.btn.white{background:var(--paper)}
.btn.pink{background:var(--pink);color:var(--paper)}
.row{display:flex;flex-wrap:wrap;gap:14px}
.status{position:absolute;top:-18px;right:18px;font-family:var(--display);font-size:1rem;padding:4px 12px;border:var(--edge);background:var(--c,var(--sun));color:var(--ink);box-shadow:0 4px 0 var(--ink);transform:rotate(3deg)}
.status.live::before{content:"";display:inline-block;width:9px;height:9px;background:currentColor;margin-right:7px;vertical-align:1px;animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:0}}

/* ---------- hero ---------- */
.hero{min-height:100vh;min-height:100svh;display:flex;flex-direction:column;justify-content:flex-end;padding-block:120px 7vh}
.hero-copy{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:24px;align-items:end}
.hero h1{font-family:var(--display);font-weight:400;font-size:clamp(2.8rem,6.6vw,6rem);line-height:.92;color:var(--paper);
  -webkit-text-stroke:clamp(6px,.9vw,10px) var(--ink);paint-order:stroke fill;text-shadow:0 clamp(5px,.7vw,9px) 0 var(--ink);transform:rotate(-2deg);transform-origin:left bottom;max-width:12ch}
.hero .lead{margin-top:20px;background:var(--paper);border:var(--edge);box-shadow:var(--lift);padding:14px 18px;max-width:46ch;font-size:1.08rem}
.hero .row{margin-top:22px}
.hint{font-family:var(--display);font-size:1rem;color:var(--paper);-webkit-text-stroke:5px var(--ink);paint-order:stroke fill;display:flex;align-items:center;gap:10px;justify-self:end}
.hint i{display:block;width:22px;height:34px;border:var(--edge);background:var(--paper);position:relative}
.hint i::after{content:"";position:absolute;left:50%;top:6px;width:5px;height:8px;margin-left:-2.5px;background:var(--ink);animation:wheel 1.4s infinite}
@keyframes wheel{0%{transform:translateY(0);opacity:1}100%{transform:translateY(12px);opacity:0}}
@media (max-width:700px){.hero-copy{grid-template-columns:1fr}.hint{display:none}}

/* ---------- about ---------- */
.facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:20px}
.fact{border:var(--edge);padding:12px 14px;background:var(--haze)}
.fact b{display:block;font-family:var(--display);font-weight:400;font-size:1.7rem;line-height:1}
.fact span{font-size:.92rem;color:var(--muted);font-weight:800}

/* ---------- games ---------- */
.game-tag{display:inline-block;font-family:var(--display);font-size:1rem;color:var(--paper);background:var(--c);border:var(--edge);padding:2px 10px;margin-bottom:12px;box-shadow:0 3px 0 var(--ink)}
.game .big-type{font-size:clamp(2.4rem,5vw,3.8rem)}
.game ul{list-style:none;display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.game li{font-weight:900;font-size:.9rem;border:2px solid var(--ink);padding:3px 10px;background:color-mix(in srgb,var(--c) 18%,#fff)}


/* ---------- live Roblox games ---------- */
.now-on{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:22px}
.now-label{font-family:var(--display);font-size:1.05rem;color:var(--paper);-webkit-text-stroke:5px var(--ink);paint-order:stroke fill}
.now-game{display:inline-flex;align-items:center;gap:10px;background:var(--paper);border:var(--edge);box-shadow:0 4px 0 var(--ink);padding:4px 12px 4px 4px;font-weight:900;text-decoration:none;transition:transform .12s,box-shadow .12s}
.now-game:hover{transform:translateY(-2px);box-shadow:0 6px 0 var(--ink)}
.now-game img{width:40px;height:40px;border:2px solid var(--ink);background:var(--haze);object-fit:cover}
.game.real{width:min(620px,100%);padding:clamp(20px,2.6vw,30px)}
.game.real .big-type{font-size:clamp(2.1rem,4vw,3.2rem);margin-bottom:8px}
.shot{display:block;aspect-ratio:5/2;max-width:100%;border:var(--edge);background:var(--haze);overflow:hidden;margin-bottom:16px;box-shadow:0 5px 0 var(--ink)}
.shot img{display:block;width:100%;height:100%;object-fit:cover;transition:transform .4s}
.shot:hover img{transform:scale(1.04)}
.live-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:14px}
.live-stats div{border:2px solid var(--ink);background:color-mix(in srgb,var(--c) 14%,#fff);padding:8px 10px;min-width:0}
.live-stats dt{font-size:.78rem;font-weight:900;color:var(--muted)}
.live-stats dd{font-family:var(--display);font-size:1.35rem;line-height:1.1;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.panel p.updated{margin-top:8px;font-size:.88rem;font-weight:800}
.play-row{margin-top:18px;align-items:center}
.playing{display:inline-flex;align-items:center;gap:8px;font-weight:900;border:var(--edge);background:var(--paper);padding:6px 12px}
.playing::before{content:"";width:10px;height:10px;background:var(--grass);border:2px solid var(--ink);animation:blink 1s steps(2) infinite}
.how{list-style:none;display:grid;gap:6px;margin-top:12px}
.game .how li{position:relative;padding:0 0 0 22px;border:0;background:none;font-size:.98rem;font-weight:800}
.how li::before{content:"";position:absolute;left:0;top:.45em;width:11px;height:11px;background:var(--c);border:2px solid var(--ink)}
.lab{width:min(560px,100%)}
.lab .game-tag{background:var(--c)}
.lab-list{list-style:none;display:grid;gap:10px;margin-top:16px}
.lab-list li{display:grid;gap:2px;border:var(--edge);border-left:12px solid var(--c);padding:10px 14px;background:var(--paper)}
.lab-list b{font-family:var(--display);font-weight:400;font-size:1.3rem;line-height:1.1}
.lab-list span{color:var(--muted);font-size:.98rem}

/* ---------- shop ---------- */
.shop-wrap{width:min(1200px,100%)}
.shop-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:end;gap:16px;margin-bottom:22px}
.shop-head .panel{width:min(560px,100%)}
.inv{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.slot{background:var(--paper);border:var(--edge);box-shadow:0 6px 0 var(--ink);padding:18px;display:grid;grid-template-columns:64px minmax(0,1fr);gap:14px;align-items:start;transition:transform .15s,box-shadow .15s}
.slot:hover{transform:translateY(-4px) rotate(-.6deg);box-shadow:0 10px 0 var(--ink)}
.slot canvas{width:64px;height:64px;border:var(--edge);background:var(--c);image-rendering:pixelated}
.slot h3{font-family:var(--display);font-weight:400;font-size:1.3rem;line-height:1.05}
.slot p{font-size:.95rem;color:var(--muted);margin-top:4px}
.slot small{display:inline-block;margin-top:8px;font-weight:900;font-size:.8rem;border:2px solid var(--ink);padding:1px 8px;background:var(--c)}
@media (max-width:980px){.inv{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:620px){.inv{grid-template-columns:1fr}}

/* ---------- contact ---------- */
.contact .panel{width:min(620px,100%)}
form{display:grid;gap:12px;margin-top:18px}
label.f{display:grid;gap:6px;font-family:var(--display);font-size:1.05rem}
input,textarea{font:inherit;font-weight:700;color:var(--ink);background:var(--haze);border:var(--edge);padding:12px 14px;width:100%;border-radius:0;outline:none}
input:focus,textarea:focus{background:var(--paper);box-shadow:0 0 0 4px var(--sun)}
textarea{min-height:120px;resize:vertical}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chips input{position:absolute;opacity:0;width:1px;height:1px}
.chips label{font-weight:900;border:var(--edge);padding:6px 12px;cursor:pointer;background:var(--paper);transition:transform .1s}
.chips label:hover{transform:translateY(-2px)}
.chips input:checked + label{background:var(--sun);box-shadow:0 4px 0 var(--ink)}
.chips input:focus-visible + label{outline:4px solid var(--pink);outline-offset:2px}
.note{font-weight:900;min-height:1.4em}

/* ---------- footer ---------- */
footer{padding-block:40px 30px}
.foot{background:var(--ink);color:var(--paper);border:var(--edge);box-shadow:0 8px 0 rgba(20,32,70,.35);padding:28px;display:flex;flex-wrap:wrap;justify-content:space-between;gap:20px;align-items:center}
.foot .big-type{font-size:clamp(2.4rem,8vw,5rem);color:var(--sun)}
.foot nav{display:flex;flex-wrap:wrap;gap:16px;font-weight:900}
.foot nav a{text-decoration:none}
.foot nav a:hover{text-decoration:underline;text-decoration-thickness:3px}
.foot small{display:block;width:100%;opacity:.7;font-weight:700}

/* ---------- loader ---------- */
#boot{position:fixed;inset:0;z-index:50;display:grid;place-items:center;background:var(--sky);transition:opacity .5s,visibility .5s}
#boot.gone{opacity:0;visibility:hidden}
.boot-in{display:grid;justify-items:center;gap:16px}
.boot-cubes{display:flex;gap:8px}
.boot-cubes i{width:22px;height:22px;background:var(--sun);border:var(--edge);animation:hop .6s ease-in-out infinite alternate}
.boot-cubes i:nth-child(2){background:var(--pink);animation-delay:.15s}
.boot-cubes i:nth-child(3){background:var(--ice);animation-delay:.3s}
@keyframes hop{to{transform:translateY(-14px) rotate(90deg)}}
.boot-in p{font-family:var(--display);font-size:1.3rem;color:var(--paper);-webkit-text-stroke:5px var(--ink);paint-order:stroke fill}

/* entrances: visible at rest, nudged in when ready */
.js .panel.rise{transition:transform .7s cubic-bezier(.2,1.4,.4,1)}
.js .panel.rise:not(.in){transform:translateY(28px) rotate(-1.5deg)}

@media (max-width:820px){.stop,.stop.right{align-items:flex-end;justify-content:center;padding-block:96px 40px}.hero{padding-block:96px 6vh}}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important;transition:none!important}
  .js .panel.rise:not(.in){transform:none}
}
</style>
</head>
<body>
<div id="boot" aria-hidden="true"><div class="boot-in"><div class="boot-cubes"><i></i><i></i><i></i></div><p>Loading worlds</p></div></div>

<canvas id="world" aria-label="A 3D voxel world with the Zixle logo, a Roblox-style character and floating islands for each Zixle game"></canvas>

<header class="bar">
  <div class="bar-in">
    <a class="logo" href="#top" aria-label="Zixle Studios, back to start"><b>ZX</b><span>Zixle Studios</span></a>
    <button class="menu-btn" id="menuBtn" aria-expanded="false" aria-controls="menu">Menu</button>
    <nav class="menu" id="menu">
      <a href="#about">About Us</a>
      <a href="#games">Games</a>
      <a href="#shop">Marketplace</a>
      <a href="#contact">Contact</a>
      <a class="play" href="#games">View Games</a>
    </nav>
  </div>
</header>

<nav class="tracker" aria-label="Islands">
  <a href="#top" data-i="0"><span>Zixle HQ</span></a>
  <a href="#about" data-i="1"><span>About</span></a>
  <a href="#games" data-i="2"><span>Snowball Fight!</span></a>
  <a href="#brainrot" data-i="3"><span>Don’t Let Brainrot Escape</span></a>
  <a href="#lab" data-i="4"><span>In the lab</span></a>
  <a href="#shop" data-i="5"><span>Marketplace</span></a>
  <a href="#contact" data-i="6"><span>Contact</span></a>
</nav>

<main class="page">
  <section class="hero wrap" id="top" data-stop="0">
    <div class="hero-copy">
      <div>
        <h1>We make Zixle games.</h1>
        <p class="lead">Original Roblox worlds with <strong>bacon characters</strong>, wild thumbnails, game passes, merch drops and updates that give you a reason to come back.</p>
        <div class="row">
          <a class="btn" href="#games">View Games</a>
          <a class="btn white" href="#about">About Us</a>
        </div>
        <div class="now-on" aria-label="Live on Roblox">
          <span class="now-label">Live on Roblox</span>
          <a class="now-game" href="https://www.roblox.com/games/106779828450798" target="_blank" rel="noopener"><img src="/api/zixle/img/10187813903/icon" alt="" width="40" height="40" loading="lazy">Snowball Fight!</a>
          <a class="now-game" href="https://www.roblox.com/games/86065026589130" target="_blank" rel="noopener"><img src="/api/zixle/img/7514162847/icon" alt="" width="40" height="40" loading="lazy">Don’t Let Brainrot Escape</a>
        </div>
      </div>
      <div class="hint" aria-hidden="true"><i></i>Scroll to fly</div>
    </div>
  </section>

  <section class="stop wrap right" id="about" data-stop="1">
    <div class="panel rise">
      <h2 class="big-type">Player worlds with a loud identity.</h2>
      <p>Zixle Studios builds its own Roblox games. Bold art, clear launches, featured modes, access passes and community merch.</p>
      <p>Every game should make sense in the first second: <strong>who the characters are, what the fight is, what you can unlock</strong>, and why it's better with friends.</p>
      <div class="facts">
        <div class="fact"><b data-live="totals.visits">1,419</b><span>Total visits</span></div>
        <div class="fact"><b data-live="group.members">18</b><span>Community members</span></div>
        <div class="fact"><b data-live="totals.games">2</b><span>Games live on Roblox</span></div>
        <div class="fact"><b data-live="totals.favorites">7</b><span>Favorites</span></div>
      </div>
      <div class="row" style="margin-top:20px"><a class="btn" href="https://www.roblox.com/communities/908132892" target="_blank" rel="noopener">Join the community</a><a class="btn white" href="#contact">Contact Zixle</a></div>
    </div>
  </section>

  <section class="stop wrap" id="games" data-stop="2">
    <article class="panel rise game real" style="--c:var(--ice)" data-universe="10187813903">
      <span class="status live" style="--c:var(--ice)">Alpha</span>
      <span class="game-tag">Live on Roblox</span>
      <a class="shot" href="https://www.roblox.com/games/106779828450798" target="_blank" rel="noopener" tabindex="-1" aria-hidden="true"><img src="/api/zixle/img/10187813903/thumb" alt="" loading="lazy" width="768" height="432"></a>
      <h2 class="big-type">Snowball Fight!</h2>
      <p>Welcome to the alpha of Snowball Fight! New features and bug fixes land every day. Find a glitch? Let us know, and leave a like to support early development.</p>
      <dl class="live-stats">
        <div><dt>Visits</dt><dd data-live="10187813903.visits">87</dd></div>
        <div><dt>Favorites</dt><dd data-live="10187813903.favorites">5</dd></div>
        <div><dt>Server size</dt><dd data-live="10187813903.maxPlayers">5</dd></div>
      </dl>
      <p class="updated">Last updated <span data-live="10187813903.updated">Jul 31, 2026</span></p>
      <div class="row play-row"><a class="btn" href="https://www.roblox.com/games/106779828450798" target="_blank" rel="noopener">Play on Roblox</a><span class="playing" data-playing="10187813903" hidden></span></div>
    </article>
  </section>

  <section class="stop wrap right" id="brainrot" data-stop="3">
    <article class="panel rise game real" style="--c:var(--pink)" data-universe="7514162847">
      <span class="status live" style="--c:var(--grass)">Live</span>
      <span class="game-tag">Live on Roblox</span>
      <a class="shot" href="https://www.roblox.com/games/86065026589130" target="_blank" rel="noopener" tabindex="-1" aria-hidden="true"><img src="/api/zixle/img/7514162847/thumb" alt="" loading="lazy" width="432" height="432"></a>
      <h2 class="big-type">DON’T LET BRAINROT ESCAPE!!</h2>
      <p>Chaos is breaking loose. Brainrots are escaping and it's your job to stop them.</p>
      <ul class="how"><li>Buy Brainrots and lock them in your base</li><li>Upgrade and mutate them to raise their value</li><li>Steal Brainrots from other players, if you're fast enough</li></ul>
      <dl class="live-stats">
        <div><dt>Visits</dt><dd data-live="7514162847.visits">1,332</dd></div>
        <div><dt>Favorites</dt><dd data-live="7514162847.favorites">2</dd></div>
        <div><dt>Server size</dt><dd data-live="7514162847.maxPlayers">8</dd></div>
      </dl>
      <p class="updated">Last updated <span data-live="7514162847.updated">Mar 28, 2026</span></p>
      <div class="row play-row"><a class="btn" href="https://www.roblox.com/games/86065026589130" target="_blank" rel="noopener">Play on Roblox</a><span class="playing" data-playing="7514162847" hidden></span></div>
    </article>
  </section>

  <section class="stop wrap" id="lab" data-stop="4">
    <div class="panel rise lab">
      <span class="game-tag" style="--c:var(--smoke)">In the lab</span>
      <h2 class="big-type">Next up from Zixle.</h2>
      <p>Ideas we're building toward. Tell us which one you want first.</p>
      <ul class="lab-list">
        <li style="--c:var(--smoke)"><b>U Got Smoked</b><span>Quick-round battles built for loud wins and thumbnail moments.</span></li>
        <li style="--c:var(--fire)"><b>Fireball Arena</b><span>Ice versus fire powers in arenas that feel like they're falling apart.</span></li>
        <li style="--c:var(--pink)"><b>Bacon City</b><span>A social world for cosmetics, hangouts, codes, shops and events.</span></li>
      </ul>
      <div class="row" style="margin-top:18px"><a class="btn white" href="#contact">Vote with an idea</a></div>
    </div>
  </section>

  <section class="stop wrap" id="shop" data-stop="5">
    <div class="shop-wrap">
      <div class="shop-head">
        <div class="panel rise">
          <h2 class="big-type">Access, merch and drops.</h2>
          <p>Unlock it in the game, flex it in the game. Join the Zixle community on Roblox to grab group rewards in our games.</p>
          <div class="row" style="margin-top:16px"><a class="btn" href="https://www.roblox.com/communities/908132892" target="_blank" rel="noopener">Join the community</a></div>
        </div>
      </div>
      <div class="inv">
        <div class="slot" style="--c:var(--sun)"><canvas width="16" height="16" data-icon="door"></canvas><div><h3>VIP doors</h3><p>Private rooms and spawns only pass holders can walk through.</p><small>Game pass</small></div></div>
        <div class="slot" style="--c:var(--ice)"><canvas width="16" height="16" data-icon="bolt"></canvas><div><h3>Early access areas</h3><p>Play new maps and modes before everyone else.</p><small>Game pass</small></div></div>
        <div class="slot" style="--c:var(--smoke)"><canvas width="16" height="16" data-icon="gem"></canvas><div><h3>Cosmetic packs</h3><p>Trails, effects and fits that make your bacon stand out.</p><small>Bundle</small></div></div>
        <div class="slot" style="--c:var(--grass)"><canvas width="16" height="16" data-icon="shirt"></canvas><div><h3>Creator merch</h3><p>Zixle avatar clothing and accessories from the studio.</p><small>Merch</small></div></div>
        <div class="slot" style="--c:var(--fire)"><canvas width="16" height="16" data-icon="clock"></canvas><div><h3>Limited event rewards</h3><p>Show up during live events for drops that never come back.</p><small>Limited</small></div></div>
        <div class="slot" style="--c:var(--pink)"><canvas width="16" height="16" data-icon="gift"></canvas><div><h3>Bacon starter bundles</h3><p>Everything a fresh bacon needs on day one.</p><small>Starter</small></div></div>
      </div>
    </div>
  </section>

  <section class="stop wrap contact" id="contact" data-stop="6">
    <div class="panel rise">
      <h2 class="big-type">Got a world idea?</h2>
      <p>Tell us about game modes, merch, access passes or community features you want next. The best ideas end up in the game.</p>
      <form id="ideaForm">
        <label class="f" for="fName">Name or Roblox username<input id="fName" name="name" required placeholder="BaconLegend_99" autocomplete="nickname"></label>
        <div class="chips" role="radiogroup" aria-label="Topic">
          <input type="radio" name="topic" id="tp1" value="Game mode" checked><label for="tp1">Game mode</label>
          <input type="radio" name="topic" id="tp2" value="Merch"><label for="tp2">Merch</label>
          <input type="radio" name="topic" id="tp3" value="Access passes"><label for="tp3">Access passes</label>
          <input type="radio" name="topic" id="tp4" value="Community"><label for="tp4">Community</label>
        </div>
        <label class="f" for="fMsg">Your idea<textarea id="fMsg" name="msg" required placeholder="Lava floor round in Snowball Chase"></textarea></label>
        <div class="row" style="align-items:center"><button class="btn pink" type="submit">Send idea</button><span class="note" id="note" role="status"></span></div>
      </form>
    </div>
  </section>

  <footer class="wrap">
    <div class="foot">
      <span class="big-type">ZIXLE</span>
      <nav aria-label="Footer"><a href="#about">About Us</a><a href="#games">Games</a><a href="#shop">Marketplace</a><a href="#contact">Contact</a><a href="https://www.roblox.com/communities/908132892" target="_blank" rel="noopener">Roblox community</a></nav>
      <small>© <span id="yr">2026</span> Zixle Studios. Original Roblox games, player access, bacon characters and marketplace drops. Not affiliated with Roblox Corporation.</small>
    </div>
  </footer>
</main>
<script>
/* ============ page UI: boot, menu, island tracker, entrances, shop icons, idea form ============ */
const CONTACT_EMAIL = ""; // put the inbox that should receive ideas here, e.g. "hello@zixlestudios.com"
(function(){
  const root = document.documentElement; root.classList.add('js');
  const $ = (s) => document.querySelector(s), $$ = (s) => [...document.querySelectorAll(s)];
  $('#yr').textContent = new Date().getFullYear();

  // boot screen -> signal the world to assemble the logo
  let went = false;
  const go = () => { if (went) return; went = true; $('#boot').classList.add('gone'); window.__zgo = true; dispatchEvent(new Event('zixle:go')); };
  addEventListener('load', () => setTimeout(go, 350));
  setTimeout(go, 2600);

  // mobile menu
  const btn = $('#menuBtn'), menu = $('#menu');
  btn.addEventListener('click', () => { const o = menu.classList.toggle('open'); btn.setAttribute('aria-expanded', o); btn.textContent = o ? 'Close' : 'Menu'; });
  menu.addEventListener('click', e => { if (e.target.closest('a')){ menu.classList.remove('open'); btn.setAttribute('aria-expanded', false); btn.textContent = 'Menu'; } });

  // island tracker follows the section nearest the middle of the screen
  const stops = $$('[data-stop]'), dots = $$('.tracker a');
  const track = () => {
    const mid = innerHeight / 2; let best = 0, bd = 1e9;
    stops.forEach((el, i) => { const r = el.getBoundingClientRect(); const d = Math.abs(r.top + r.height / 2 - mid); if (d < bd){ bd = d; best = i; } });
    dots.forEach((d, i) => d.classList.toggle('on', i === best));
  };
  addEventListener('scroll', track, {passive:true}); track();

  // panels pop in once when they reach the screen (visible by default if this never runs)
  if ('IntersectionObserver' in window){
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); } }), {threshold:.2});
    $$('.panel.rise').forEach(p => io.observe(p));
  } else $$('.panel.rise').forEach(p => p.classList.add('in'));

  // 8x8 pixel icons for the marketplace slots
  const ICONS = {
    door: ['..kkkk..','.kyyyyk.','.kyuuyk.','.kyuuyk.','.kyuuyk.','.kyuwyk.','.kyuuyk.','kkkkkkkk'],
    bolt: ['....kkk.','...kyyk.','..kyyk..','.kyyyykk','kkkyyk..','..kyk...','.kyk....','.kk.....'],
    gem:  ['..kkkk..','.kwppwk.','kwppppwk','kkkkkkkk','.kppppk.','..kppk..','...kk...','........'],
    shirt:['.kk..kk.','kggkkggk','kggggggk','.kggggk.','.kgwwgk.','.kggggk.','.kggggk.','.kkkkkk.'],
    clock:['..kkkk..','.kwwwwk.','kwwkwwwk','kwwkwwwk','kwwkkkwk','kwwwwwwk','.kwwwwk.','..kkkk..'],
    gift: ['.kk..kk.','..kkkk..','kkkkkkkk','kppkkppk','kppkkppk','kppkkppk','kppkkppk','kkkkkkkk'],
  };
  const PAL = {k:'#142046', w:'#ffffff', y:'#ffd23a', u:'#8a5cff', p:'#ff4f9a', g:'#4fd26b'};
  $$('canvas[data-icon]').forEach(cv => {
    const g = cv.getContext('2d'), rows = ICONS[cv.dataset.icon]; if (!rows) return;
    rows.forEach((row, y) => [...row].forEach((ch, x) => { if (PAL[ch]){ g.fillStyle = PAL[ch]; g.fillRect(x * 2, y * 2, 2, 2); } }));
  });


  // live Roblox numbers, served by /api/zixle on this same site
  const fmtN = (n) => (typeof n === 'number') ? n.toLocaleString('en-US') : n;
  const fmtD = (iso) => { const d = new Date(iso); return isNaN(d) ? '' : d.toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'}); };
  const applyLive = (d) => {
    const byId = {}; (d.games || []).forEach(g => { byId[g.universeId] = g; });
    $$('[data-live]').forEach(el => {
      const [a, b] = el.dataset.live.split('.');
      let v = a === 'totals' ? (d.totals || {})[b] : a === 'group' ? (d.group || {})[b] : (byId[a] || {})[b];
      if (b === 'updated' && v) v = fmtD(v);
      if (v !== undefined && v !== null && v !== '') el.textContent = fmtN(v);
    });
    $$('[data-playing]').forEach(el => { const g = byId[el.dataset.playing]; const n = g ? g.playing : 0; el.hidden = !(n > 0); if (n > 0) el.textContent = fmtN(n) + ' playing now'; });
    window.__zlive = d; dispatchEvent(new CustomEvent('zixle:live', {detail:d}));
  };
  const refresh = () => fetch('/api/zixle', {headers:{Accept:'application/json'}}).then(r => r.ok ? r.json() : null).then(d => { if (d && d.games) applyLive(d); }).catch(() => {});
  if (/^https?:$/.test(location.protocol)){ refresh(); setInterval(() => { if (!document.hidden) refresh(); }, 60000); }
  // a game image that can't load leaves its colored frame instead of a broken icon
  $$('img').forEach(img => { const hide = () => { img.style.visibility = 'hidden'; }; if (img.complete && !img.naturalWidth) hide(); else img.addEventListener('error', hide, {once:true}); });

  // idea form
  $('#ideaForm').addEventListener('submit', e => {
    e.preventDefault();
    const f = new FormData(e.target), note = $('#note');
    if (!CONTACT_EMAIL){ note.textContent = 'The ideas inbox opens soon. Check back!'; return; }
    const subject = `[Zixle idea] ${f.get('topic')} from ${f.get('name')}`;
    location.href = `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(f.get('msg') + '\n\n- ' + f.get('name'))}`;
    note.textContent = 'Opening your email app with your idea filled in.';
  });
})();
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
if (!window.THREE) return;
try {
/* ============ ZIXLE WORLD — part 1: core, voxel builder, logo, avatar ============ */

const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;
const SMALL = matchMedia('(max-width: 820px)').matches;
const COARSE = matchMedia('(pointer: coarse)').matches;
const INK = 0x142046;

const Z = {}; window.__Z = Z; // shared world state
function startWorld(){
  if (!window.THREE) throw new Error('three missing');
  const canvas = document.getElementById('world');
  const renderer = new THREE.WebGLRenderer({canvas, antialias:!SMALL, alpha:true, powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, SMALL ? 1.25 : 1.75));
  renderer.shadowMap.enabled = !SMALL;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xc9f0ff, 70, 300);
  const camera = new THREE.PerspectiveCamera(SMALL ? 58 : 42, 1, 0.1, 900);
  Object.assign(Z, {renderer, scene, camera, canvas, t:0});

  const hemi = new THREE.HemisphereLight(0xdff4ff, 0x6fbf73, 0.95);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff1d6, 1.05);
  sun.castShadow = !SMALL;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera; sc.left = -46; sc.right = 46; sc.top = 46; sc.bottom = -46; sc.near = 1; sc.far = 220;
  sun.shadow.bias = -0.0008;
  scene.add(sun, sun.target);
  Z.sun = sun;

  const resize = () => {
    const w = innerWidth, h = innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  addEventListener('resize', resize); resize();
}

/* ---------- color helpers ---------- */
const col = (hex) => new THREE.Color(hex);
function jitter(hex, amt){ const c = col(hex); const hsl = {}; c.getHSL(hsl); c.setHSL(hsl.h, hsl.s, Math.min(1, Math.max(0, hsl.l + (Math.random() - .5) * amt))); return c; }

/* ---------- voxel builder: many boxes -> one mesh + one ink outline (inverted hull) ---------- */
const UNIT = () => new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
class Vox {
  constructor(){ this.boxes = []; }
  box(x, y, z, w, h, d, color, opts = {}){ this.boxes.push({x, y, z, w, h, d, color: color instanceof THREE.Color ? color : col(color), glow: opts.glow || 0, line: opts.line !== false}); return this; }
  build(outline = 0.14){
    const base = UNIT(); const bp = base.attributes.position.array, bn = base.attributes.normal.array; const vc = bp.length / 3;
    const n = this.boxes.length;
    const pos = new Float32Array(n * vc * 3), nor = new Float32Array(n * vc * 3), clr = new Float32Array(n * vc * 3);
    const lines = this.boxes.filter(b => b.line);
    const opos = new Float32Array(lines.length * vc * 3);
    let o = 0, oo = 0;
    for (const b of this.boxes){
      const c = b.glow ? b.color.clone().multiplyScalar(1 + b.glow) : b.color;
      for (let i = 0; i < vc; i++){
        pos[o] = bp[i*3] * b.w + b.x; pos[o+1] = bp[i*3+1] * b.h + b.y; pos[o+2] = bp[i*3+2] * b.d + b.z;
        nor[o] = bn[i*3]; nor[o+1] = bn[i*3+1]; nor[o+2] = bn[i*3+2];
        clr[o] = c.r; clr[o+1] = c.g; clr[o+2] = c.b; o += 3;
      }
      if (b.line) for (let i = 0; i < vc; i++){
        opos[oo] = bp[i*3] * (b.w + outline) + b.x; opos[oo+1] = bp[i*3+1] * (b.h + outline) + b.y; opos[oo+2] = bp[i*3+2] * (b.d + outline) + b.z; oo += 3;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('color', new THREE.BufferAttribute(clr, 3));
    const mesh = new THREE.Mesh(g, Z.mats.vox);
    mesh.castShadow = true; mesh.receiveShadow = true;
    const grp = new THREE.Group(); grp.add(mesh);
    if (lines.length){
      const og = new THREE.BufferGeometry(); og.setAttribute('position', new THREE.BufferAttribute(opos, 3));
      grp.add(new THREE.Mesh(og, Z.mats.ink));
    }
    return grp;
  }
}
function makeMats(){
  Z.mats = {
    vox: new THREE.MeshStandardMaterial({vertexColors:true, roughness:.85, metalness:0, flatShading:true}),
    ink: new THREE.MeshBasicMaterial({color:INK, side:THREE.BackSide}),
  };
}
/* a single outlined primitive (spheres, orbs) */
function outlined(geo, mat, thick = 0.08){
  const g = new THREE.Group();
  const m = new THREE.Mesh(geo, mat); m.castShadow = true; g.add(m);
  const ol = new THREE.Mesh(geo, Z.mats.ink); ol.scale.setScalar(1 + thick); g.add(ol);
  g.userData.core = m; return g;
}

/* ---------- floating island: grassy cap + stepped underside ---------- */
function island(v, cx, cy, cz, r, top, dirt, stone, opts = {}){
  const cell = 2, seed = opts.seed || 1;
  const rnd = (i, j, k) => { const s = Math.sin(i * 127.1 + j * 311.7 + k * 74.7 + seed * 19.3) * 43758.5453; return s - Math.floor(s); };
  const topFn = opts.topColor || (() => top);
  let layer = 0, rr = r;
  while (rr > 1.2){
    const y = cy - layer * cell * .9;
    for (let i = -Math.ceil(rr); i <= Math.ceil(rr); i++) for (let j = -Math.ceil(rr); j <= Math.ceil(rr); j++){
      const d = Math.hypot(i, j) + rnd(i, j, layer) * 1.1;
      if (d > rr) continue;
      const x = cx + i * cell, z = cz + j * cell;
      if (layer === 0){
        v.box(x, y, z, cell, cell * .9, cell, jitter(topFn(i, j, x, z), .07));
      } else {
        const c = layer < 2 ? dirt : stone;
        v.box(x, y, z, cell, cell * .9, cell, jitter(c, .08));
      }
    }
    layer++; rr = r - layer * (opts.taper || 1.55) - rnd(layer, 2, 3) * .6;
  }
  return cy + cell * .45; // walkable top y
}
function tree(v, x, y, z, s = 1, leaf = '#3fbf5f'){
  v.box(x, y + 1.5 * s, z, .9 * s, 3 * s, .9 * s, '#8a5a32');
  v.box(x, y + 3.8 * s, z, 3.2 * s, 2 * s, 3.2 * s, jitter(leaf, .06));
  v.box(x, y + 5.2 * s, z, 2 * s, 1.2 * s, 2 * s, jitter(leaf, .1));
}
function pine(v, x, y, z, s = 1){
  v.box(x, y + 1 * s, z, .8 * s, 2 * s, .8 * s, '#7a4a2a');
  [[3.6, 2.6], [2.7, 4], [1.8, 5.3], [.9, 6.4]].forEach(([w, h], k) => {
    v.box(x, y + h * s, z, w * s, 1.2 * s, w * s, jitter('#2f9e5a', .06));
    v.box(x, y + (h + .65) * s, z, w * s * .8, .25 * s, w * s * .8, '#f4fbff', {line:false});
  });
}

/* ---------- voxel ZIXLE logo (instanced, assembles on load, reacts to cursor) ---------- */
const GLYPHS = {
  Z:['11111','00001','00010','00100','01000','10000','11111'],
  I:['111','010','010','010','010','010','111'],
  X:['10001','10001','01010','00100','01010','10001','10001'],
  L:['10000','10000','10000','10000','10000','10000','11111'],
  E:['11111','10000','10000','11110','10000','10000','11111'],
};
function buildLogo(){
  const cells = []; let cx = 0;
  for (const ch of 'ZIXLE'){
    const g = GLYPHS[ch];
    g.forEach((row, ry) => [...row].forEach((bit, rx) => { if (bit === '1') cells.push({x: cx + rx, y: 6 - ry}); }));
    cx += g[0].length + 1;
  }
  const width = cx - 1, S = 1;
  const list = [];
  cells.forEach(c => {
    for (let layer = 0; layer < 2; layer++){
      const t = c.y / 6;
      const front = col('#ffd23a').lerp(col('#ff8a1a'), 1 - t);
      const back = col('#e0560f').lerp(col('#a8320b'), 1 - t);
      list.push({home: new THREE.Vector3((c.x - width / 2 + .5) * S, (c.y - 3) * S, -layer * S), color: layer ? back : front});
    }
  });
  const geo = new THREE.BoxGeometry(S, S, S);
  const mat = new THREE.MeshStandardMaterial({roughness:.55, metalness:.05, flatShading:true});
  const fill = new THREE.InstancedMesh(geo, mat, list.length);
  const ink = new THREE.InstancedMesh(geo, Z.mats.ink, list.length);
  fill.castShadow = true;
  list.forEach((b, i) => {
    fill.setColorAt(i, b.color);
    const a = Math.random() * Math.PI * 2, e = (Math.random() - .3) * 1.4, R = 40 + Math.random() * 40;
    b.start = new THREE.Vector3(Math.cos(a) * R, Math.sin(e) * R + 10, Math.sin(a) * R * .6 - 20);
    b.spin = new THREE.Vector3(Math.random() * 6, Math.random() * 6, Math.random() * 6);
    b.delay = (b.home.x / width + .5) * .7 + Math.random() * .25;
    b.off = new THREE.Vector3(); b.vel = new THREE.Vector3();
  });
  fill.instanceColor.needsUpdate = true;
  const grp = new THREE.Group(); grp.add(fill, ink);
  Z.logo = {grp, fill, ink, list, width, born: -1, dummy: new THREE.Object3D()};
  return grp;
}
const easeOutBack = (t) => { const c1 = 1.9, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
function updateLogo(dt, pointerLocal){
  const L = Z.logo; if (!L) return;
  const now = Z.t, d = L.dummy;
  const age = L.born < 0 ? 0 : (now - L.born);
  L.list.forEach((b, i) => {
    let p = REDUCE ? 1 : Math.min(1, Math.max(0, (age - b.delay) / 1.25));
    // pointer push + spring
    if (pointerLocal && p >= 1){
      const dx = b.home.x - pointerLocal.x, dy = b.home.y - pointerLocal.y; const dist = Math.hypot(dx, dy);
      if (dist < 4.2){ const f = (1 - dist / 4.2); b.vel.z += f * f * 28 * dt; b.vel.x += dx / (dist + .01) * f * 6 * dt; b.vel.y += dy / (dist + .01) * f * 6 * dt; }
    }
    b.vel.addScaledVector(b.off, -42 * dt); b.vel.multiplyScalar(Math.pow(.02, dt));
    b.off.addScaledVector(b.vel, dt);
    const e = easeOutBack(p);
    d.position.lerpVectors(b.start, b.home, e).add(b.off);
    const r = (1 - Math.min(1, p * 1.2));
    d.rotation.set(b.spin.x * r + b.off.y * .25, b.spin.y * r + b.off.x * .25, b.spin.z * r);
    d.scale.setScalar(.25 + .75 * Math.min(1, p * 1.5));
    d.updateMatrix(); L.fill.setMatrixAt(i, d.matrix);
    d.scale.multiplyScalar(1.13); d.updateMatrix(); L.ink.setMatrixAt(i, d.matrix);
  });
  L.fill.instanceMatrix.needsUpdate = true; L.ink.instanceMatrix.needsUpdate = true;
}
function logoBurst(local){
  if (!Z.logo) return;
  Z.logo.list.forEach(b => {
    const dx = b.home.x - local.x, dy = b.home.y - local.y, dist = Math.hypot(dx, dy) + .5;
    const k = 26 / dist;
    b.vel.x += dx / dist * k; b.vel.y += dy / dist * k + 4; b.vel.z += 10 + Math.random() * 14;
  });
}

/* ---------- R6-style bacon-hair avatar ---------- */
function buildAvatar(){
  const skin = '#ffcc99', shirt = '#ff4f9a', pants = '#2c3e9e';
  const part = (w, h, d, color, px, py, pz) => { const v = new Vox(); v.box(0, -h / 2, 0, w, h, d, color); const g = v.build(.16); const piv = new THREE.Group(); piv.position.set(px, py, pz); piv.add(g); return piv; };
  const av = new THREE.Group();
  const legL = part(1, 2, 1, pants, -.5, 2, 0), legR = part(1, 2, 1, pants, .5, 2, 0);
  const torsoV = new Vox(); torsoV.box(0, 3, 0, 2, 2, 1, shirt); torsoV.box(0, 3.25, .51, .9, .5, .05, '#ffffff', {line:false}); // little chest logo
  const torso = torsoV.build(.16);
  const armL = part(1, 2, 1, skin, -1.5, 4, 0), armR = part(1, 2, 1, skin, 1.5, 4, 0);
  const head = new THREE.Group(); head.position.set(0, 4, 0);
  const hv = new Vox();
  hv.box(0, .7, 0, 1.3, 1.3, 1.3, skin);
  hv.box(-.28, .85, .66, .16, .26, .04, '#142046', {line:false}); hv.box(.28, .85, .66, .16, .26, .04, '#142046', {line:false});
  hv.box(0, .45, .66, .55, .1, .04, '#142046', {line:false}); hv.box(-.3, .52, .66, .1, .12, .04, '#142046', {line:false}); hv.box(.3, .52, .66, .1, .12, .04, '#142046', {line:false});
  // bacon hair: streaky brown slabs
  const hair = ['#7a3e1a', '#c8743c', '#7a3e1a', '#c8743c', '#7a3e1a'];
  hair.forEach((c, k) => { hv.box(-.56 + k * .28, 1.45 + (k % 2) * .08, -.05, .28, .35, 1.42, c); });
  hv.box(-.68, 1.05, -.1, .12, .8, 1.3, '#7a3e1a'); hv.box(.68, 1.05, -.1, .12, .8, 1.3, '#c8743c');
  hv.box(0, 1.0, -.7, 1.42, 1.1, .14, '#9a5228');
  head.add(hv.build(.12));
  av.add(legL, legR, torso, armL, armR, head);
  Z.avatar = {g: av, legL, legR, armL, armR, head, wave: 0};
  return av;
}
function updateAvatar(dt, look){
  const A = Z.avatar; if (!A) return;
  const t = Z.t;
  A.g.position.y = A.baseY + Math.abs(Math.sin(t * 2.2)) * .12;
  A.armL.rotation.x = Math.sin(t * 2.2) * .25;
  A.wave = Math.max(0, A.wave - dt);
  const w = A.wave > 0 ? Math.sin(A.wave * 14) * .35 : 0;
  A.armR.rotation.z = A.wave > 0 ? 2.6 + w : -Math.sin(t * 2.2) * .1;
  A.armR.rotation.x = A.wave > 0 ? 0 : -Math.sin(t * 2.2) * .25;
  if (look){ A.head.rotation.y += (look.x * .7 - A.head.rotation.y) * .1; A.head.rotation.x += (-look.y * .35 - A.head.rotation.x) * .1; }
}

/* ============ ZIXLE WORLD — part 2: islands, props, camera flight, loop ============ */
const ISL = {
  hq:     new THREE.Vector3(0, 0, 0),
  snow:   new THREE.Vector3(84, -2, -14),
  brain:  new THREE.Vector3(166, 2, 8),
  fire:   new THREE.Vector3(250, -1, -10),
  smoke:  new THREE.Vector3(296, 6, -46),
  bacon:  new THREE.Vector3(336, 1, -4),
  shop:   new THREE.Vector3(402, 3, -8),
};
const anim = []; // per-frame updaters
const arc = (a, b, u, h, out) => out.lerpVectors(a, b, u).setY(out.y + Math.sin(Math.PI * u) * h);


/* a framed billboard that shows a real game image (same-origin proxy, so WebGL can use it) */
function billboard(x, groundY, z, w, h, lift, yaw, src, frame){
  const g = new THREE.Group(), v = new Vox(), t = .7, cy = h / 2 + lift;
  v.box(0, cy + h / 2 + t / 2, 0, w + 2 * t, t, .8, frame); v.box(0, cy - h / 2 - t / 2, 0, w + 2 * t, t, .8, frame);
  v.box(-w / 2 - t / 2, cy, 0, t, h, .8, frame); v.box(w / 2 + t / 2, cy, 0, t, h, .8, frame);
  v.box(-w / 3, lift / 2 - .3, -.2, .9, lift + .2, .9, '#5a6075'); v.box(w / 3, lift / 2 - .3, -.2, .9, lift + .2, .9, '#5a6075');
  v.box(0, cy, -.36, w, h, .1, '#142046', {line:false});
  g.add(v.build(.16));
  const mat = new THREE.MeshBasicMaterial({color:0x9fdcff});
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat); screen.position.set(0, cy, .41); g.add(screen);
  if (src && /^https?:$/.test(location.protocol)) new THREE.TextureLoader().load(src, tex => { tex.anisotropy = 4; mat.map = tex; mat.color.set(0xffffff); mat.needsUpdate = true; });
  g.position.set(x, groundY, z); g.rotation.y = yaw; Z.scene.add(g);
  return g;
}
function brainrotGuy(){
  const v = new Vox();
  v.box(0, 1.55, 0, 1.6, 2.1, 1, '#c98d4f'); v.box(-.2, 2.95, 0, 1.2, .8, 1, '#c98d4f'); v.box(.5, 2.75, 0, .5, .5, 1, '#d9a066');
  v.box(-.36, 2.05, .51, .42, .48, .06, '#ffffff', {line:false}); v.box(.36, 2.05, .51, .42, .48, .06, '#ffffff', {line:false});
  v.box(-.32, 1.98, .56, .18, .24, .04, '#142046', {line:false}); v.box(.4, 1.98, .56, .18, .24, .04, '#142046', {line:false});
  v.box(0, 1.05, .62, .52, .8, .26, '#ff5fa2', {line:false});
  v.box(-.45, .25, 0, .42, .5, .42, '#8a5a32'); v.box(.45, .25, 0, .42, .5, .42, '#8a5a32');
  return v.build(.12);
}

function buildWorld(){
  const S = Z.scene;

  /* --- HQ island: logo, avatar, trees --- */
  { const v = new Vox(), c = ISL.hq;
    const top = island(v, c.x, c.y, c.z, 8.6, '#57d36b', '#a8693a', '#7d8aa6', {seed:1});
    tree(v, c.x - 12, top, c.z - 8, 1.1); tree(v, c.x + 13, top, c.z - 6, .9); tree(v, c.x + 9, top, c.z + 9, .7, '#5fd06a');
    v.box(c.x + 6, top + .9, c.z + 4, 1.8, 1.8, 1.8, '#c8874a'); v.box(c.x + 7.4, top + .7, c.z + 6, 1.4, 1.4, 1.4, '#d99a58');
    v.box(c.x - 2, top + .15, c.z + 10, 6, .3, 2, '#ffd23a'); // spawn pad
    S.add(v.build());
    const logo = buildLogo(); logo.scale.setScalar(1.25); logo.position.set(c.x, c.y + 15, c.z - 3); S.add(logo); Z.logoPos = logo.position;
    const av = buildAvatar(); av.position.set(c.x - 9, top, c.z + 5); av.rotation.y = .35; Z.avatar.baseY = top; S.add(av);
  }

  /* --- Snowball Chase: snowy island, pines, forts, snowballs flying --- */
  { const v = new Vox(), c = ISL.snow;
    const top = island(v, c.x, c.y, c.z, 9.2, '#f2fbff', '#9fc9e8', '#6f86a8', {seed:2, topColor:(i, j) => ((i * 7 + j * 3) % 5 === 0 ? '#d6efff' : '#f4fbff')});
    [[-13,-8,1.2],[-9,-12,.9],[12,-10,1.1],[15,-4,.8],[-15,2,.85],[10,10,.75]].forEach(([x,z,s]) => pine(v, c.x + x, top, c.z + z, s));
    const fort = (x, z, rot) => { for (let k = -2; k <= 2; k++){ const dx = rot ? 0 : k * 1.6, dz = rot ? k * 1.6 : 0; v.box(c.x + x + dx, top + .8, c.z + z + dz, 1.6, 1.6, 1.6, jitter('#e6f6ff', .05)); if (Math.abs(k) < 2) v.box(c.x + x + dx, top + 2.2, c.z + z + dz, 1.4, 1.2, 1.4, jitter('#f6fcff', .04)); } };
    fort(-7, 2, true); fort(7, 0, true);
    S.add(v.build());
    const ballMat = new THREE.MeshStandardMaterial({color:0xffffff, roughness:.7, flatShading:true});
    const sg = new THREE.IcosahedronGeometry(.75, 1);
    for (let k = 0; k < 6; k++){ const b = outlined(sg, ballMat, .1); b.position.set(c.x - 9 + (k % 3) * 1.2, top + .7 + Math.floor(k / 3) * .9, c.z + 5 + (k % 2) * .8); S.add(b); }
    const A = new THREE.Vector3(c.x - 7, top + 3.2, c.z + 2), B = new THREE.Vector3(c.x + 7, top + 3.2, c.z);
    [0, .5].forEach(ph => { const b = outlined(sg, ballMat, .1); S.add(b); anim.push(t => { const u = (t * .45 + ph) % 1; const fw = Math.floor(t * .45 + ph) % 2 === 0; arc(fw ? A : B, fw ? B : A, u, 6, b.position); b.rotation.x = t * 6; }); });
    billboard(c.x + 1, top, c.z - 13, 17, 9.6, 3, .32, '/api/zixle/img/10187813903/thumb', '#39c6ff');
    Z.snowFx = particles(c, 26, 22, 900, 0xffffff, .32, -2.2);
  }


  /* --- DON'T LET BRAINROT ESCAPE!!: rainbow BASE, sirens, brainrots running for the edge --- */
  { const v = new Vox(), c = ISL.brain;
    const top = island(v, c.x, c.y, c.z, 9.2, '#57d36b', '#a8693a', '#7d8aa6', {seed:7, topColor:(i, j) => (j === 1 ? '#8f96ad' : ((i + j) % 6 === 0 ? '#4cc463' : '#5ad86f'))});
    const bx = c.x - 6, bz = c.z - 4;
    ['#ff3b3b', '#ff8a1a', '#ffd23a', '#4fd26b', '#39a0ff', '#8a5cff'].forEach((col, k) => v.box(bx, top + .6 + k * 1.2, bz, 8, 1.2, 6, col));
    v.box(bx, top + 7.45, bz, 8.6, .5, 6.6, '#142046');
    v.box(bx, top + 1.7, bz + 3.02, 2.4, 3.4, .1, '#142046', {line:false});
    v.box(bx - 2.7, top + 4.4, bz + 3.02, 1.6, 1.4, .1, '#bfe9ff', {line:false}); v.box(bx + 2.7, top + 4.4, bz + 3.02, 1.6, 1.4, .1, '#bfe9ff', {line:false});
    v.box(bx, top + 9.1, bz + 1.6, 6.6, 2.7, .5, '#142046');
    [[9, 7], [12, -2], [-12, 6], [6, 12], [-3, 11]].forEach(([x, z]) => v.box(c.x + x, top + .9, c.z + z, 3.4, 1.8, 1, '#9aa3bd'));
    S.add(v.build());
    const cv = document.createElement('canvas'); cv.width = 256; cv.height = 96; const g2 = cv.getContext('2d');
    g2.fillStyle = '#ffd23a'; g2.fillRect(0, 0, 256, 96); g2.fillStyle = '#142046'; g2.font = '68px "Lilita One", Impact, "Arial Black", sans-serif'; g2.textAlign = 'center'; g2.textBaseline = 'middle'; g2.fillText('BASE', 128, 54);
    const signTex = new THREE.CanvasTexture(cv);
    const sign = new THREE.Mesh(new THREE.PlaneGeometry(6, 2.25), new THREE.MeshBasicMaterial({map:signTex})); sign.position.set(bx, top + 9.1, bz + 1.88); S.add(sign);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { g2.fillStyle = '#ffd23a'; g2.fillRect(0, 0, 256, 96); g2.fillStyle = '#142046'; g2.fillText('BASE', 128, 54); signTex.needsUpdate = true; });
    const sirenMat = new THREE.MeshBasicMaterial({color:0xff2a2a});
    [-3.2, 3.2].forEach(dx => { const m = outlined(new THREE.BoxGeometry(1, .9, 1), sirenMat, .16); m.position.set(bx + dx, top + 8.2, bz + 2); S.add(m); });
    anim.push(t => sirenMat.color.setHex(Math.sin(t * 11) > 0 ? 0xff2a2a : 0x6a0f0f));
    billboard(c.x + 6, top, c.z - 12, 11, 11, 3, -.3, '/api/zixle/img/7514162847/thumb', '#ff4f9a');
    for (let k = 0; k < 6; k++){
      const b = brainrotGuy(); S.add(b);
      const a = -.25 + k * .3, sp = .13 + (k % 3) * .03, ph = k / 6;
      anim.push(t => { const u = (t * sp + ph) % 1; const r = 1 + u * 15; const fall = Math.max(0, u - .82) * 60;
        b.position.set(bx + Math.sin(a) * r, top + Math.abs(Math.sin(t * 11 + k)) * .55 - fall * fall * .05, bz + 3.4 + Math.cos(a) * r);
        b.rotation.set(0, a, Math.sin(t * 11 + k) * .15); b.visible = u < .97; });
    }
    const coinGeo = new THREE.CylinderGeometry(.8, .8, .25, 10), coinMat = new THREE.MeshStandardMaterial({color:0xffd23a, roughness:.35, metalness:.3, flatShading:true});
    [[4, 3], [7, -3], [-1, 7], [10, 3], [2, -6]].forEach(([x, z], k) => { const m = outlined(coinGeo, coinMat, .14); S.add(m); anim.push(t => { m.position.set(c.x + x, top + 1.2 + Math.sin(t * 2 + k) * .3, c.z + z); m.rotation.set(Math.PI / 2, 0, t * 2.5 + k); }); });
  }

  /* --- U Got Smoked: night-stone arena, pillars, rising smoke --- */
  { const v = new Vox(), c = ISL.smoke;
    const top = island(v, c.x, c.y, c.z, 8.4, '#5b3fa8', '#3a2a6e', '#2a2350', {seed:3, topColor:(i, j) => ((i + j) & 1 ? '#6a4bc0' : '#ff4f9a')});
    for (let k = 0; k < 8; k++){ const a = k / 8 * Math.PI * 2, x = c.x + Math.cos(a) * 13, z = c.z + Math.sin(a) * 13;
      v.box(x, top + 2.5, z, 1.8, 5, 1.8, '#30285c'); v.box(x, top + 5.4, z, 2.4, .8, 2.4, '#ff4f9a', {glow:.4}); }
    v.box(c.x, top + .5, c.z, 6, 1, 6, '#ffd23a'); v.box(c.x, top + 1.2, c.z, 4, .4, 4, '#ff6a1a');
    S.add(v.build());
    const ring = new THREE.Mesh(new THREE.TorusGeometry(15.5, .35, 6, 48), new THREE.MeshBasicMaterial({color:0xff4f9a}));
    ring.rotation.x = Math.PI / 2; ring.position.set(c.x, top + 6.6, c.z); S.add(ring);
    const smokeMat = new THREE.MeshStandardMaterial({color:0xcbb8ff, roughness:1, transparent:true, opacity:.85, flatShading:true});
    const pg = new THREE.IcosahedronGeometry(1, 0);
    for (let k = 0; k < 16; k++){ const m = new THREE.Mesh(pg, smokeMat.clone()); S.add(m);
      const a0 = Math.random() * 6.28, r0 = 1 + Math.random() * 3, sp = .5 + Math.random() * .5, ph = Math.random();
      anim.push(t => { const u = (t * .18 * sp + ph) % 1; m.position.set(c.x + Math.cos(a0 + u * 2) * r0 * (1 + u), top + 1.5 + u * 14, c.z + Math.sin(a0 + u * 2) * r0 * (1 + u)); m.scale.setScalar(.6 + u * 2.6); m.material.opacity = .9 * (1 - u); m.rotation.set(u * 3, u * 2, 0); }); }
    anim.push(t => { ring.rotation.z = t * .4; ring.position.y = top + 6.6 + Math.sin(t * 1.5) * .3; });
  }

  /* --- Fireball Arena: half ice, half lava, two duelling orbs --- */
  { const v = new Vox(), c = ISL.fire;
    const top = island(v, c.x, c.y, c.z, 9, '#fff', '#5a3a2a', '#3a2a28', {seed:4, topColor:(i, j, x) => { if (i < 0) return (i + j) % 3 === 0 ? '#9fe6ff' : '#d8f6ff'; return (i * 3 + j * 5) % 7 === 0 ? '#ffb020' : ((i + j) % 4 === 0 ? '#ff5a12' : '#3b2220'); }});
    for (let k = 0; k < 7; k++){ const x = c.x - 6 - Math.random() * 10, z = c.z - 10 + Math.random() * 20; v.box(x, top + 1.5, z, 1.6, 3 + Math.random() * 3, 1.6, jitter('#bff0ff', .06)); }
    for (let k = 0; k < 6; k++){ const x = c.x + 6 + Math.random() * 10, z = c.z - 10 + Math.random() * 20; v.box(x, top + 1, z, 2, 2 + Math.random() * 2, 2, '#2b1a18'); v.box(x, top + 2.6, z, 1, .6, 1, '#ff8a1a', {glow:.5, line:false}); }
    S.add(v.build());
    const og = new THREE.IcosahedronGeometry(1.8, 1);
    const fireOrb = outlined(og, new THREE.MeshBasicMaterial({color:0xff7a1a}), .08);
    const iceOrb = outlined(og, new THREE.MeshBasicMaterial({color:0x7fe3ff}), .08);
    const fl = new THREE.PointLight(0xff7a1a, 2.2, 30, 2), il = new THREE.PointLight(0x6fdcff, 1.8, 30, 2);
    fireOrb.add(fl); iceOrb.add(il); S.add(fireOrb, iceOrb);
    anim.push(t => { const a = t * .9; fireOrb.position.set(c.x + Math.cos(a) * 6, top + 6 + Math.sin(t * 2) * 1.2, c.z + Math.sin(a) * 6); iceOrb.position.set(c.x - Math.cos(a) * 6, top + 6 - Math.sin(t * 2) * 1.2, c.z - Math.sin(a) * 6); fireOrb.rotation.y = iceOrb.rotation.x = t * 1.4; const s = 1 + Math.sin(t * 9) * .05; fireOrb.scale.setScalar(s); });
    Z.emberFx = particles(new THREE.Vector3(c.x + 8, c.y, c.z), 16, 20, 500, 0xff8a1a, .38, 2.8);
  }

  /* --- Bacon City: blocky skyline, glowing windows, a giant floating bacon strip --- */
  { const v = new Vox(), c = ISL.bacon;
    const top = island(v, c.x, c.y, c.z, 9.4, '#7b84a8', '#a8693a', '#7d8aa6', {seed:5, topColor:(i, j) => (i % 4 === 0 || j % 4 === 0 ? '#ffd9ec' : '#6f7899')});
    const pal = ['#ff7ab6', '#8a6bff', '#39c6ff', '#ffd23a', '#ff9a5a', '#5fd38a'];
    let n = 0;
    for (let i = -3; i <= 3; i++) for (let j = -3; j <= 3; j++){
      if ((i % 2 === 0) || (j % 2 === 0) || Math.hypot(i, j) > 3.6) continue;
      const h = 5 + ((i * 13 + j * 7 + 20) % 9) * 1.4, x = c.x + i * 4, z = c.z + j * 4, color = pal[n++ % pal.length];
      v.box(x, top + h / 2, z, 3.2, h, 3.2, color);
      for (let y = 1.6; y < h - .8; y += 1.6){ v.box(x - .7, top + y, z + 1.62, .7, .7, .06, '#fff3b0', {glow:.4, line:false}); v.box(x + .7, top + y, z + 1.62, .7, .7, .06, (y * 7 | 0) % 3 ? '#fff3b0' : '#4a3f7a', {glow:.4, line:false}); }
      v.box(x, top + h + .25, z, 3.4, .5, 3.4, '#142046', {line:false});
    }
    S.add(v.build());
    const bacon = new THREE.Group(); const bv = new Vox();
    for (let k = 0; k < 12; k++){ const y = Math.sin(k * .9) * 1.1; bv.box(k * 1.2 - 7, y, 0, 1.25, 1, .5, '#d9473a'); bv.box(k * 1.2 - 7, y + .95, 0, 1.25, .9, .5, '#ffc4a8'); bv.box(k * 1.2 - 7, y + 1.85, 0, 1.25, .9, .5, '#c7392e'); }
    bacon.add(bv.build(.18)); bacon.position.set(c.x, top + 24, c.z - 2); S.add(bacon);
    anim.push(t => { bacon.rotation.y = Math.sin(t * .6) * .5; bacon.position.y = top + 24 + Math.sin(t * 1.2) * 1.2; });
  }

  /* --- Marketplace: golden VIP door, crates, gem, coins --- */
  { const v = new Vox(), c = ISL.shop;
    const top = island(v, c.x, c.y, c.z, 8.2, '#ffe28a', '#a8693a', '#7d8aa6', {seed:6, topColor:(i, j) => ((i + j) & 1 ? '#ffe9a8' : '#ffd23a')});
    v.box(c.x - 4, top + 5, c.z - 4, 1.8, 10, 1.8, '#ffc21a'); v.box(c.x + 4, top + 5, c.z - 4, 1.8, 10, 1.8, '#ffc21a'); v.box(c.x, top + 10.6, c.z - 4, 10, 1.8, 1.8, '#ffc21a');
    v.box(c.x, top + 12, c.z - 4, 3, 1.2, 1.2, '#ff4f9a');
    [[-11,4],[-9,7],[10,5],[12,1],[-12,-2]].forEach(([x, z], k) => { const s = 1.6 + (k % 2) * .6; v.box(c.x + x, top + s / 2, c.z + z, s, s, s, '#b8742e'); v.box(c.x + x, top + s / 2, c.z + z + s / 2, s * .7, s * .18, .05, '#7a4a1a', {line:false}); });
    S.add(v.build());
    const portal = new THREE.Mesh(new THREE.PlaneGeometry(6.2, 8.8), new THREE.MeshBasicMaterial({color:0x8a5cff}));
    portal.position.set(c.x, top + 4.6, c.z - 4); S.add(portal);
    const gem = outlined(new THREE.OctahedronGeometry(2, 0), new THREE.MeshStandardMaterial({color:0xff4f9a, roughness:.3, flatShading:true, emissive:0x5a0b2e}), .08);
    gem.position.set(c.x + 7, top + 8, c.z + 2); S.add(gem);
    const coinGeo = new THREE.CylinderGeometry(1, 1, .3, 10), coinMat = new THREE.MeshStandardMaterial({color:0xffd23a, roughness:.35, metalness:.3, flatShading:true});
    const coins = [0, 1, 2, 3, 4].map(k => { const m = outlined(coinGeo, coinMat, .12); S.add(m); return m; });
    anim.push(t => { gem.rotation.y = t * 1.2; gem.position.y = top + 8 + Math.sin(t * 1.6) * .8; portal.material.color.setHSL(.72 + Math.sin(t * 1.5) * .06, .9, .62);
      coins.forEach((m, k) => { const a = t * .7 + k * 1.256; m.position.set(c.x + Math.cos(a) * 9, top + 4 + Math.sin(t * 2 + k) * 1.2, c.z + Math.sin(a) * 9); m.rotation.set(Math.PI / 2, 0, t * 3 + k); }); });
  }

  /* --- drifting voxel clouds --- */
  for (let k = 0; k < 18; k++){
    const v = new Vox(); const n = 3 + (k % 3);
    for (let i = 0; i < n; i++){ const w = 6 + ((k * 7 + i * 3) % 5) * 1.5; v.box(i * 4.2 - n * 2, (i % 2) * 1.5, ((i * 5) % 3) - 1, w, 3 + (i % 2) * 1.5, 5, '#ffffff'); }
    const cl = v.build(.2); cl.traverse(o => { if (o.isMesh) o.castShadow = false; });
    const x0 = -60 + k * 30, y = 22 + (k * 13 % 5) * 5, z = -50 - (k * 17 % 4) * 18 + (k % 5 === 0 ? 70 : 0);
    cl.position.set(x0, y, z); Z.scene.add(cl);
    anim.push(t => { cl.position.x = x0 + ((t * (1 + k % 3) * .6) % 60) - 30; });
  }
}

/* particle cloud around a point (snow falls, embers rise) */
function particles(center, rx, ry, count, color, size, speed){
  if (SMALL) count = Math.floor(count * .45);
  const g = new THREE.BufferGeometry(), p = new Float32Array(count * 3), seed = new Float32Array(count);
  for (let i = 0; i < count; i++){ p[i*3] = center.x + (Math.random() - .5) * rx * 2; p[i*3+1] = center.y + Math.random() * ry * 2 - 4; p[i*3+2] = center.z + (Math.random() - .5) * rx * 2; seed[i] = Math.random() * 6.28; }
  g.setAttribute('position', new THREE.BufferAttribute(p, 3));
  const m = new THREE.Points(g, new THREE.PointsMaterial({color, size, sizeAttenuation:true, transparent:true, opacity:.95, depthWrite:false}));
  Z.scene.add(m);
  const lo = center.y - 4, hi = center.y + ry * 2 - 4;
  anim.push((t, dt) => { for (let i = 0; i < count; i++){ let y = p[i*3+1] + speed * dt * (.6 + (seed[i] % 1)); if (y < lo) y = hi; if (y > hi) y = lo; p[i*3+1] = y; p[i*3] += Math.sin(t + seed[i]) * dt * .6; } g.attributes.position.needsUpdate = true; });
  return m;
}

/* ---------- camera stops (one per page section) ---------- */
// side: +1 subject sits right of the panel, -1 left of it, 0 centered
const STOPS = [
  {p:() => ISL.hq.clone().add(new THREE.Vector3(0, 6, 0)),       yaw:-.12, dist:52,  h:6,  side:.72},
  {p:() => ISL.hq.clone().add(new THREE.Vector3(-7, 4.5, 4)),     yaw:-.55, dist:26,  h:5,  side:-1},
  {p:() => ISL.snow.clone().add(new THREE.Vector3(0, 5, -3)),     yaw:.32,  dist:48,  h:10, side:1},
  {p:() => ISL.brain.clone().add(new THREE.Vector3(0, 5, -2)),    yaw:-.3,  dist:48,  h:12, side:-1},
  {p:() => new THREE.Vector3(292, 4, -22),                        yaw:.05,  dist:150, h:34, side:1},
  {p:() => ISL.shop.clone().add(new THREE.Vector3(0, 7, 0)),      yaw:.12,  dist:48,  h:9,  side:.7},
  {p:() => ISL.hq.clone().add(new THREE.Vector3(0, 9, 0)),        yaw:-.62, dist:54,  h:20, side:1},
];
function stopView(s){
  const cam = Z.camera, subj = s.p();
  const portrait = cam.aspect < 1;
  const dist = s.dist * (portrait ? 1.55 : cam.aspect < 1.3 ? 1.2 : 1);
  const pos = new THREE.Vector3(subj.x + Math.sin(s.yaw) * dist, subj.y + s.h, subj.z + Math.cos(s.yaw) * dist);
  const halfH = dist * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)), halfW = halfH * cam.aspect;
  const right = new THREE.Vector3(Math.cos(s.yaw), 0, -Math.sin(s.yaw));
  const tgt = subj.clone();
  if (SMALL || portrait) tgt.y -= halfH * (s === STOPS[0] ? .36 : .38);
  else tgt.addScaledVector(right, -s.side * halfW * .42);
  return {pos, tgt};
}
let stopEls = [], stopCenters = [];
function measure(){
  stopEls = [...document.querySelectorAll('[data-stop]')];
  stopCenters = stopEls.map(el => { const r = el.getBoundingClientRect(); return r.top + scrollY + r.height / 2 - innerHeight / 2; });
  stopCenters[0] = 0;
}
function scrollStop(){
  const y = scrollY, c = stopCenters; if (!c.length) return 0;
  if (y <= c[0]) return 0;
  for (let i = 0; i < c.length - 1; i++) if (y < c[i + 1]) return i + (y - c[i]) / (c[i + 1] - c[i]);
  return c.length - 1;
}
const smooth = (t) => t * t * (3 - 2 * t);
const camPos = new THREE.Vector3(), camTgt = new THREE.Vector3(), wantPos = new THREE.Vector3(), wantTgt = new THREE.Vector3();
let views = [];
function desired(f){
  const i = Math.min(Math.floor(f), views.length - 2), u = smooth(Math.min(1, Math.max(0, f - i)));
  const a = views[i], b = views[i + 1] || views[i];
  wantPos.lerpVectors(a.pos, b.pos, u); wantTgt.lerpVectors(a.tgt, b.tgt, u);
  const span = a.tgt.distanceTo(b.tgt);
  wantPos.y += Math.sin(Math.PI * u) * Math.min(26, span * .16);
}

/* ---------- pointer ---------- */
const ndc = new THREE.Vector2(9, 9), ray = new THREE.Raycaster(), plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), hit = new THREE.Vector3();
let pointerLocal = null;
function pointerToLogo(){
  if (!Z.logo || ndc.x > 2) return null;
  ray.setFromCamera(ndc, Z.camera);
  plane.constant = -Z.logo.grp.position.z;
  if (!ray.ray.intersectPlane(plane, hit)) return null;
  return Z.logo.grp.worldToLocal(hit.clone());
}

/* ---------- main loop ---------- */
function run(){
  makeMats(); startWorld(); buildWorld();
  measure(); views = STOPS.map(stopView);
  desired(scrollStop()); camPos.copy(wantPos); camTgt.copy(wantTgt);
  addEventListener('resize', () => { measure(); views = STOPS.map(stopView); });
  addEventListener('load', () => { measure(); views = STOPS.map(stopView); });
  addEventListener('pointermove', e => { ndc.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1); }, {passive:true});
  document.addEventListener('pointerleave', () => ndc.set(9, 9));
  Z.canvas.addEventListener('pointerdown', e => {
    ndc.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    const p = pointerToLogo();
    if (p && Math.abs(p.x) < Z.logo.width / 2 + 3 && Math.abs(p.y) < 7) logoBurst(p);
    if (Z.avatar) Z.avatar.wave = 1.4;
  });
  let last = performance.now(), visible = true;
  document.addEventListener('visibilitychange', () => { visible = !document.hidden; last = performance.now(); });
  const frame = (now) => {
    requestAnimationFrame(frame);
    if (!visible) return;
    const dt = Math.min(.05, (now - last) / 1000); last = now; Z.t += dt;
    desired(scrollStop());
    const k = (REDUCE || window.__zsnap) ? 1 : 1 - Math.pow(.004, dt);
    camPos.lerp(wantPos, k); camTgt.lerp(wantTgt, k);
    // a little parallax sway from the mouse
    const sway = (ndc.x > 2 || COARSE) ? 0 : 1;
    Z.camera.position.copy(camPos).add(new THREE.Vector3(ndc.x * 1.6 * sway, ndc.y * .9 * sway, 0));
    Z.camera.lookAt(camTgt);
    Z.sun.position.copy(camTgt).add(new THREE.Vector3(38, 64, 30)); Z.sun.target.position.copy(camTgt);
    pointerLocal = pointerToLogo();
    updateLogo(dt, pointerLocal);
    if (Z.logo){ Z.logo.grp.rotation.y = Math.sin(Z.t * .5) * .08 + (ndc.x < 2 ? ndc.x * .12 : 0); Z.logo.grp.position.y = ISL.hq.y + 15 + Math.sin(Z.t * 1.1) * .4; }
    updateAvatar(dt, ndc.x < 2 ? ndc : null);
    for (const fn of anim) fn(Z.t, dt);
    Z.renderer.render(Z.scene, Z.camera);
  };
  requestAnimationFrame(frame);
}

run();
const born = () => { if (Z.logo && Z.logo.born < 0) Z.logo.born = Z.t + .1; };
addEventListener('zixle:go', born); if (window.__zgo) born();
} catch (err) { console.error(err); const c = document.getElementById('world'); if (c) c.style.display = 'none'; }
})();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HOME_HTML)

@app.get("/health")
def health():
    return {"ok": True, "studio": "Zixle Studios", "type": "original Roblox games", "theme": "based-style studio landing"}

# ---------------------------------------------------------------------------
# Live Roblox data for the homepage (public Roblox APIs, cached)
# ---------------------------------------------------------------------------
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

GROUP_ID = 908132892
FEATURED_UNIVERSES = [10187813903, 7514162847]  # Snowball Fight! (alpha), DON'T LET BRAINROT ESCAPE!!
LIVE_TTL_SECONDS = 60
_live_cache = {"at": 0.0, "data": None}
_live_lock = threading.Lock()
_img_cache = {}
_HEADERS = {"User-Agent": "ZixleStudiosSite/1.0 (+https://www.zixlestudios.com)", "Accept": "application/json"}


def _get_json(url):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=6) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_live():
    ids = ",".join(str(u) for u in FEATURED_UNIVERSES)
    urls = {
        "games": f"https://games.roblox.com/v1/games?universeIds={ids}",
        "votes": f"https://games.roblox.com/v1/games/votes?universeIds={ids}",
        "icons": f"https://thumbnails.roblox.com/v1/games/icons?universeIds={ids}&returnPolicy=PlaceHolder&size=512x512&format=Png&isCircular=false",
        "thumbs": f"https://thumbnails.roblox.com/v1/games/multiget/thumbnails?universeIds={ids}&countPerUniverse=5&defaults=true&size=768x432&format=Png&isCircular=false",
        "group": f"https://groups.roblox.com/v1/groups/{GROUP_ID}",
    }
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = {key: pool.submit(_get_json, url) for key, url in urls.items()}
        raw = {}
        for key, fut in futures.items():
            try:
                raw[key] = fut.result()
            except Exception:
                raw[key] = None
    if not raw["games"]:
        raise RuntimeError("Roblox games API unavailable")

    votes = {v["id"]: v for v in (raw["votes"] or {}).get("data", [])}
    icons = {i["targetId"]: i.get("imageUrl") for i in (raw["icons"] or {}).get("data", [])}
    thumbs = {t["universeId"]: [x.get("imageUrl") for x in t.get("thumbnails", []) if x.get("imageUrl")] for t in (raw["thumbs"] or {}).get("data", [])}
    by_id = {g["id"]: g for g in raw["games"].get("data", [])}

    games = []
    for uid in FEATURED_UNIVERSES:
        g = by_id.get(uid)
        if not g:
            continue
        v = votes.get(uid, {})
        games.append({
            "universeId": uid,
            "placeId": g.get("rootPlaceId"),
            "name": g.get("name"),
            "description": g.get("description"),
            "playing": g.get("playing") or 0,
            "visits": g.get("visits") or 0,
            "favorites": g.get("favoritedCount") or 0,
            "maxPlayers": g.get("maxPlayers"),
            "created": g.get("created"),
            "updated": g.get("updated"),
            "upVotes": v.get("upVotes"),
            "downVotes": v.get("downVotes"),
            "url": f"https://www.roblox.com/games/{g.get('rootPlaceId')}",
            "icon": icons.get(uid),
            "thumbnails": thumbs.get(uid, []),
        })

    group = raw["group"] or {}
    return {
        "fetchedAt": datetime.now(timezone.utc).isoformat(),
        "group": {
            "id": GROUP_ID,
            "name": group.get("name", "*ZIXLE* Studios"),
            "members": group.get("memberCount"),
            "url": f"https://www.roblox.com/communities/{GROUP_ID}",
        },
        "totals": {
            "games": len(games),
            "visits": sum(g["visits"] for g in games),
            "playing": sum(g["playing"] for g in games),
            "favorites": sum(g["favorites"] for g in games),
        },
        "games": games,
    }


def _live():
    now = time.time()
    with _live_lock:
        if _live_cache["data"] and now - _live_cache["at"] < LIVE_TTL_SECONDS:
            return _live_cache["data"]
    try:
        data = _fetch_live()
    except Exception:
        if _live_cache["data"]:
            return _live_cache["data"]  # serve the last good copy
        raise
    with _live_lock:
        _live_cache.update(at=now, data=data)
    return data


@app.get("/api/zixle")
def zixle_live():
    try:
        data = _live()
    except Exception:
        raise HTTPException(status_code=503, detail="Roblox data is unavailable right now")
    return JSONResponse(data, headers={"Cache-Control": "public, max-age=30, s-maxage=60, stale-while-revalidate=300"})


@app.get("/api/zixle/img/{universe_id}/{kind}")
def zixle_image(universe_id: int, kind: str):
    if universe_id not in FEATURED_UNIVERSES or kind not in ("icon", "thumb"):
        raise HTTPException(status_code=404, detail="Unknown image")
    try:
        game = next(g for g in _live()["games"] if g["universeId"] == universe_id)
    except Exception:
        raise HTTPException(status_code=503, detail="Roblox data is unavailable right now")
    url = game["icon"] if kind == "icon" else (game["thumbnails"][0] if game["thumbnails"] else game["icon"])
    if not url or not url.startswith("https://tr.rbxcdn.com/"):
        raise HTTPException(status_code=404, detail="No image yet")
    body = _img_cache.get(url)
    if body is None:
        req = urllib.request.Request(url, headers={"User-Agent": _HEADERS["User-Agent"]})
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read()
        if len(_img_cache) > 16:
            _img_cache.clear()
        _img_cache[url] = body
    return Response(content=body, media_type="image/png", headers={"Cache-Control": "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800"})
