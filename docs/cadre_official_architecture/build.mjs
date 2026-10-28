import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "/Users/kieugiangbien/.codex/skills/anatomy/kit/iso-kit.mjs";
import { kitScript } from "/Users/kieugiangbien/.codex/skills/anatomy/scripts/inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const LIGHT = process.argv.includes("--light");
const W = 1160, H = 730;
const PARTS = {
  "Đế và khung": ["Bàn đế", "4 chân giảm chấn", "Mặt bàn gia công", "Rãnh viền", "Vít góc", "Thước chuẩn", "Vạch nhỏ", "Vạch lớn", "2 ray dẫn hướng", "2 thanh giằng", "4 chặn hành trình", "Bệ mô-đun"],
  "Nguồn ngữ cảnh": ["Cassette active file", "Cuộn left-context", "2 con lăn cấp liệu", "Thanh ép băng", "Magazine repo pool", "4 thẻ file", "Lược snippet", "Máng Channel A", "Cửa file path", "Bàn ghép prompt"],
  "Generator và cổng": ["Vỏ generator dùng chung", "Nắp máy", "Trục truyền", "Rotor", "Moay-ơ", "8 khe tản nhiệt", "Khe draft", "Khe refine", "2 con lăn draft", "Cầu so dòng", "Cam copy", "Cửa bypass", "Lò xo hồi", "Máng trả draft"],
  "Intent và target API": ["Vỏ intent parser", "Lược receiver", "Lược callee", "Lược token stem", "Ray phân tích", "Giá repo pool", "3 ngăn file", "Đầu quét", "Xe quét chữ ký", "Thẻ target API"],
  "Ghép prompt và budget": ["Khay raw chunks", "Máng CSSI tùy chọn", "Kênh target-spec", "Khay file path", "Băng left-context suffix", "Cữ headroom", "Mặt số slack", "Kim budget", "Cung vạch 1536/2304", "Ray trả prompt"],
  "Đầu ra": ["Máng completion", "Khay kết quả", "Thẻ completion", "Chốt giữ thẻ"]
};
const partCount = Object.values(PARTS).reduce((n, a) => n + a.length, 0);

const DECK = {x:-244,y:-148,w:492,d:296,r:10};
const ACTIVE = {x:-214,y:-75,w:48,d:54,r:4};
const POOL = {x:-213,y:20,w:54,d:58,r:4};
const INTAKE = {x:-142,y:-54,w:48,d:112,r:4};
const ENGINE = {x:-82,y:-65,w:77,d:130,r:7};
const GATE = {x:13,y:-61,w:52,d:58,r:5};
const INTENT = {x:73,y:40,w:57,d:49,r:5};
const RESOLVER = {x:146,y:39,w:67,d:62,r:5};
const COMPOSER = {x:119,y:-99,w:91,d:48,r:5};
const OUTPUT = {x:202,y:-21,w:43,d:78,r:5};
const bounds = [
  ...k.boxCorners(DECK,-17,0), ...k.boxCorners(ENGINE,0,65),
  ...k.boxCorners(ACTIVE,0,27), ...k.boxCorners(POOL,0,26),
  ...k.boxCorners(GATE,0,31), ...k.boxCorners(INTENT,0,34),
  ...k.boxCorners(RESOLVER,0,41), ...k.boxCorners(COMPOSER,0,33),
  ...k.boxCorners(OUTPUT,0,18)
];
const P = k.fitProjection(bounds,W,H,{pad:44,headroom:4,azimuth:43,elevation:30});
const S = (p,o) => k.solidSvg(p,o || {});
const L = (d,o) => k.lineSvg(d,o || {});
const D = (p,o) => k.dotsSvg(p,o || {});
const at = (x,y,z) => k.iso([x,y,z],P);
const esc = (s) => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");

function fasteners(plan,z,inset) {
  const rr = k.corners(plan,inset || 3).map(([x,y])=>k.ring(x,y,1.5,z,P,16)).join("");
  const dots = k.corners(plan,inset || 3).map(([x,y])=>at(x,y,z));
  return L(rr,{tone:"lo"}) + D(dots,{size:0.45,tone:"lo"});
}
function seam(plan,z,by) {
  return L(k.planOutline(k.insetPlan(plan,by || 2.5),z,P,6),{tone:"lo"});
}
function box(plan,h,tone) {
  return S(k.slabOf(plan,0,h,P,8,1.1),{tone:tone || "mid",crease:"lo"}) +
    seam(plan,h,3) + fasteners(plan,h,4);
}
function rule(x1,x2,y,z,tone) {
  return L(k.lineOnTop([x1,y],[x2,y],z,P),{tone:tone || "faint"});
}
function ruledCard(plan,z,count,tone) {
  let html = S(k.slabOf(plan,z,1.6,P,4,0.3),{tone:tone || "hi",crease:"faint"});
  for(let i=1;i<=count;i++) html += rule(plan.x+3,plan.x+plan.w-3,plan.y+plan.d*i/(count+1),z+1.6,"faint");
  return html;
}
function tray(plan,n) {
  let html = S(k.slabOf(plan,0,5,P,8,1),{tone:"mid",crease:"lo"});
  for(const y of [plan.y+1.5,plan.y+plan.d-5.5])
    html += S(k.slabOf({x:plan.x+2,y:y,w:plan.w-4,d:4,r:1},5,4,P,4,0.5),{tone:"mid"});
  for(let i=n-1;i>=0;i--){
    const p={x:plan.x+6+i*2.3,y:plan.y+7+i*2,w:plan.w-15,d:plan.d-17,r:1};
    html += ruledCard(p,5.2+i*1.8,3,i===0?"hi":"mid");
  }
  return html + fasteners(plan,5,3);
}
function strip(plan,z,n) {
  let html=S(k.slabOf(plan,z,1.2,P,4,0.4),{tone:"lo",crease:"none"});
  const gap=(plan.w-8)/Math.max(1,n-1);
  for(let i=0;i<n;i++){
    const x=plan.x+4+gap*i;
    html+=L(k.lineOnTop([x,plan.y+2],[x,plan.y+plan.d-2],z+1.2,P),{tone:i%2?"faint":"lo"});
  }
  return html;
}
function conveyor(plan,n) {
  let html=S(k.slabOf(plan,1.2,1.4,P,4,0.5),{tone:"lo",crease:"none"});
  for(const y of [plan.y,plan.y+plan.d-1.5])
    html+=S(k.slabOf({x:plan.x,y:y,w:plan.w,d:1.5,r:0.6},2.3,3.3,P,3,0.4),{tone:"mid"});
  for(let i=0;i<n;i++){
    const x=plan.x+4+i*(plan.w-8)/Math.max(1,n-1);
    html+=S(k.slabOf({x:x-1.1,y:plan.y+2,w:2.2,d:plan.d-4,r:0.8},2.1,1.4,P,3,0.25),{tone:"hi",crease:"none"});
  }
  return html;
}
function cable(points,tone,cls) {
  const screen=points.map(p=>k.iso(p,P));
  const d=k.pathOf(screen);
  return "<g class=\"iso-wire "+cls+"\" data-tone=\""+tone+"\"><path class=\"iso-line\" data-tone=\""+tone+"\" d=\""+d+"\"/></g>";
}

function base() {
  let html="";
  for(const [x,y] of k.corners(DECK,18)){
    html+=S(k.cylinder(x,y,8.2,-17,4,P,28),{tone:"lo"});
    html+=S(k.cylinder(x,y,5.8,-13,5,P,28),{tone:"mid"});
    html+=L(k.sideArc(x,y,5.8,-8,P,14),{tone:"faint"});
  }
  html+=S(k.slabOf(DECK,-8,8,P,10,2),{tone:"mid",crease:"lo"});
  html+=L(k.planOutline(k.insetPlan(DECK,7),0,P,10),{tone:"lo"})+fasteners(DECK,0,12);
  const ruler={x:-218,y:122,w:438,d:8,r:1.5};
  html+=S(k.slabOf(ruler,0,0.8,P,6,0.2),{tone:"lo",crease:"none"});
  const ticks=k.topTicks(ruler.x+3,ruler.x+ruler.w-3,4,5,ruler.y+0.8,0.8,[1.4,2.8],P);
  html+=L(ticks.minor,{tone:"faint"})+L(ticks.major,{tone:"lo"});
  html+=k.faceTextSvg(k.topMatrix([-205,111],0.8,P,"x"),"CADRE",{size:4,tone:"mid",anchor:"start"});
  return html;
}
function frameTracks() {
  let html="";
  const rails=[
    {x:-222,y:-129,w:444,d:4,r:1},{x:-222,y:125,w:444,d:4,r:1},
    {x:-228,y:-118,w:4,d:236,r:1},{x:224,y:-118,w:4,d:236,r:1}
  ];
  for(const p of rails) html+=S(k.slabOf(p,0,4,P,4,0.4),{tone:"mid"});
  for(const x of [-160,-80,0,80,160]) for(const y of [-124,120])
    html+=S(k.slabOf({x:x-2.3,y:y-2.5,w:4.6,d:5,r:1},4,3,P,4,0.3),{tone:"lo"});
  html+=conveyor({x:-162,y:-19,w:74,d:11,r:2},4);
  html+=conveyor({x:-160,y:45,w:73,d:11,r:2},4);
  html+=conveyor({x:-95,y:-2,w:30,d:13,r:2},3);
  html+=conveyor({x:-8,y:-28,w:27,d:13,r:2},3);
  html+="<g class=\"route-bypass\">"+conveyor({x:61,y:-83,w:141,d:12,r:2},7)+"</g>";
  html+=conveyor({x:128,y:62,w:25,d:11,r:2},3);
  html+=conveyor({x:113,y:-116,w:94,d:11,r:2},5);
  html+=conveyor({x:-63,y:-116,w:176,d:11,r:2},8);
  html+="<g class=\"route-final\">"+conveyor({x:-6,y:100,w:210,d:12,r:2},9)+"</g>";
  html+=cable([[64,-61,5],[81,-77,5],[104,-77,5]],"faint","route-bypass");
  html+=cable([[64,-32,5],[64,22,5],[76,45,5]],"faint","route-refine");
  html+=cable([[130,65,7],[144,65,7]],"faint","route-refine");
  html+=cable([[180,39,7],[180,-74,7],[170,-92,7]],"faint","route-refine");
  html+=cable([[119,-100,7],[75,-109,7],[-40,-109,7],[-44,-61,7]],"faint","route-refine");
  html+=cable([[-5,-47,8],[8,14,8],[89,100,8],[202,100,8]],"faint","route-final");
  return html;
}
function inputs() {
  let html=tray(ACTIVE,2)+tray(POOL,4);
  html+=S(k.cylinder(-205,-60,8.8,7,8,P,40),{tone:"mid"});
  html+=S(k.cylinder(-205,-60,2.2,15,2,P,24),{tone:"hi"});
  html+=L(k.knurl(-205,-60,8.8,7,15,28,P),{tone:"lo"});
  html+=S(k.cylinder(-180,-60,5.6,7,5,P,28),{tone:"mid"});
  html+=S(k.cylinder(-188,-51,2.2,7,6,P,20),{tone:"lo"});
  html+=S(k.cylinder(-188,-69,2.2,7,6,P,20),{tone:"lo"});
  html+=S(k.slabOf({x:-211,y:-43,w:38,d:4,r:1},15,2,P,4,0.3),{tone:"hi"});
  html+=S(k.slabOf({x:-208,y:25,w:43,d:7,r:2},22,2.2,P,5,0.4),{tone:"mid"});
  html+=S(k.cylinder(-203,28,2.4,23.8,4,P,20),{tone:"hi"});
  html+=S(k.cylinder(-170,28,2.4,23.8,4,P,20),{tone:"hi"});
  html+=strip({x:-207,y:34,w:40,d:31,r:1},23.5,6);
  html+=box(INTAKE,10);
  for(let i=0;i<4;i++){
    const p={x:INTAKE.x+6+i*8,y:INTAKE.y+12,w:6,d:37,r:1};
    html+=S(k.slabOf(p,10.5,1.8,P,3,0.3),{tone:i===0?"hi":"mid",crease:"faint"});
    html+=rule(p.x+1,p.x+p.w-1,p.y+12,12.3)+rule(p.x+1,p.x+p.w-1,p.y+24,12.3);
  }
  html+=S(k.cylinder(-101,0,4.2,10,4,P,28),{tone:"mid"});
  html+=L(k.knurl(-101,0,4.2,10,14,20,P),{tone:"lo"});
  return html;
}
function engine() {
  let html=box(ENGINE,52);
  const roof={x:ENGINE.x+7,y:ENGINE.y+9,w:ENGINE.w-14,d:ENGINE.d-18,r:5};
  html+=S(k.slabOf(roof,52,6,P,8,1),{tone:"lo",crease:"faint"});
  const cx=-43,cy=-3;
  html+=S(k.cylinder(cx,cy,18,58,4,P,48,0.5),{tone:"mid"});
  html+=L(k.ring(cx,cy,14.8,62,P,48),{tone:"lo"});
  html+=S(k.cylinder(cx,cy,3.8,62,3.3,P,28),{tone:"hi"});
  for(let i=0;i<8;i++){
    const a=2*Math.PI*i/8;
    html+=L(k.lineOnTop([cx+Math.cos(a)*4,cy+Math.sin(a)*4],[cx+Math.cos(a+0.24)*14,cy+Math.sin(a+0.24)*14],62,P),{tone:"lo"});
  }
  for(let i=0;i<8;i++){
    const x=ENGINE.x+12+i*7.4;
    for(const y of [ENGINE.y+3,ENGINE.y+ENGINE.d-5])
      html+=S(k.slabOf({x:x,y:y,w:3.4,d:2,r:0.6},17,20,P,3,0.35),{tone:"lo",crease:"none"});
  }
  for(const y of [-43,20]){
    const p={x:ENGINE.x+9,y:y,w:ENGINE.w-18,d:24,r:2};
    html+=S(k.slabOf(p,12,1.3,P,4,0.3),{tone:"lo",crease:"none"});
    html+=L(k.planOutline(k.insetPlan(p,2),13.3,P,4),{tone:"faint"});
    for(let i=0;i<5;i++) html+=L(k.lineOnTop([p.x+5+i*9,p.y+4],[p.x+5+i*9,p.y+p.d-4],13.3,P),{tone:"lo"});
  }
  for(const x of [-69,-17]){
    html+=S(k.cylinder(x,-66,3.7,20,4,P,24),{tone:"hi"});
    html+=S(k.cylinder(x,65,3.7,20,4,P,24),{tone:"hi"});
  }
  const port={x:-77,y:-7,w:67,d:12,r:2};
  html+=S(k.slabOf(port,7,2.4,P,4,0.4),{tone:"lo"});
  html+=S(k.slabOf({...port,y:43},7,2.4,P,4,0.4),{tone:"lo"});
  return html;
}
function gate() {
  let html=box(GATE,18);
  for(const x of [25,53]){
    html+=S(k.cylinder(x,-34,5.6,18,3,P,32),{tone:"hi"});
    html+=L(k.ring(x,-34,3.7,21,P,24),{tone:"lo"});
    html+=D([at(x,-34,21)],{size:0.55,tone:"lo"});
  }
  const pivot=[42,-14], center=at(pivot[0],pivot[1],22).join(" ");
  html+=S(k.cylinder(pivot[0],pivot[1],4.6,18,4,P,28),{tone:"lo"});
  html+=L(k.ring(pivot[0],pivot[1],3.3,22,P,24),{tone:"hi"});
  html+="<g data-part=\"gate-arm\" transform=\"rotate(-12 "+center+")\">";
  html+=L(k.lineOnTop(pivot,[pivot[0]+15,pivot[1]-11],22.4,P),{tone:"hi",free:"end"});
  html+=D([at(pivot[0]+15,pivot[1]-11,22.4)],{size:1.4,tone:"hi"})+"</g>";
  const tooth={x:22,y:-8,w:39,d:5,r:1};
  html+=S(k.slabOf(tooth,20,2,P,4,0.4),{tone:"lo"});
  for(let i=0;i<6;i++) html+=S(k.slabOf({x:tooth.x+3+i*6.5,y:tooth.y+0.5,w:1.2,d:4,r:0.4},22,2.3,P,3,0.2),{tone:"hi",crease:"none"});
  return html;
}
function intentAndSpec() {
  let html=box(INTENT,22);
  for(let row=0;row<3;row++){
    const y=INTENT.y+11+row*12, rail={x:INTENT.x+6,y:y,w:INTENT.w-12,d:3.5,r:1};
    html+=S(k.slabOf(rail,22,2,P,4,0.4),{tone:"lo"});
    for(let j=0;j<5;j++){
      const x=rail.x+4+j*8;
      html+=S(k.slabOf({x:x,y:y-2,w:1.7,d:7.5,r:0.5},24,3.5,P,3,0.3),{tone:j===row+1?"hi":"mid",crease:"none"});
      html+=D([at(x+0.85,y+1.6,27.5)],{size:0.35,tone:"lo"});
    }
  }
  html+=L(k.planOutline(k.insetPlan(INTENT,4),22,P,5),{tone:"lo"});
  html+=box(RESOLVER,29);
  for(let i=0;i<3;i++){
    const y=RESOLVER.y+8+i*15,p={x:RESOLVER.x+7,y:y,w:RESOLVER.w-14,d:11,r:1.6};
    html+=S(k.slabOf(p,7+i*5,4,P,4,0.5),{tone:i===1?"lo":"mid"});
    html+=L(k.planOutline(k.insetPlan(p,2),11+i*5,P,4),{tone:"faint"});
    html+=D([at(p.x+4,y+5,11+i*5)],{size:0.5,tone:"hi"});
  }
  const scan={x:RESOLVER.x+5,y:RESOLVER.y+3,w:5,d:RESOLVER.d-6,r:1.2};
  html+=S(k.slabOf(scan,29,3,P,4,0.4),{tone:"lo"});
  html+=S(k.cylinder(scan.x+2.5,scan.y+8,3.1,32,2.5,P,24),{tone:"hi"});
  html+=L(k.ring(scan.x+2.5,scan.y+8,2.1,34.5,P,20),{tone:"lo"});
  const sp={x:RESOLVER.x+34,y:RESOLVER.y+6,w:25,d:39,r:1.2};
  html+="<g data-part=\"target-spec\">"+S(k.slabOf(sp,30,1.8,P,4,0.3),{tone:"hi",lit:true});
  for(const y of [9,17,25]) html+=rule(sp.x+3,sp.x+sp.w-3,sp.y+y,31.8,"lo");
  html+="</g>"+S(k.slabOf({x:sp.x+3,y:sp.y+2,w:7,d:2.5,r:0.5},31.8,0.6,P,3),{tone:"lo",crease:"none"});
  return html;
}
function composerAndOutput() {
  let html=box(COMPOSER,20);
  const a={x:COMPOSER.x+6,y:COMPOSER.y+5,w:COMPOSER.w-12,d:9,r:1.2};
  const b={x:COMPOSER.x+6,y:COMPOSER.y+17,w:COMPOSER.w-12,d:7,r:1.2};
  const spec={x:COMPOSER.x+6,y:COMPOSER.y+27,w:COMPOSER.w-12,d:7,r:1.2};
  html+=S(k.slabOf(a,20,1.4,P,4,0.3),{tone:"lo",crease:"none"});
  html+=strip({x:a.x+3,y:a.y+1,w:a.w-6,d:a.d-2,r:0.5},21.4,7);
  html+="<g data-part=\"cssi-channel\">"+S(k.slabOf(b,20,1.4,P,4,0.3),{tone:"lo",crease:"none"});
  html+=strip({x:b.x+3,y:b.y+1,w:b.w-6,d:b.d-2,r:0.5},21.4,5)+"</g>";
  html+=S(k.slabOf(spec,20,1.4,P,4,0.3),{tone:"mid",crease:"none"});
  html+=L(k.lineOnTop([spec.x+3,spec.y+3.5],[spec.x+spec.w-3,spec.y+3.5],21.4,P),{tone:"hi"});
  const dx=COMPOSER.x+72,dy=COMPOSER.y+38;
  html+=S(k.cylinder(dx,dy,12,20,3,P,44,0.5),{tone:"mid"});
  html+=L(k.ring(dx,dy,9.8,23,P,40),{tone:"lo"});
  let arc="";
  for(let i=0;i<=10;i++){
    const a=(-135+270*i/10)*Math.PI/180,r1=i%5===0?8.2:8.7,r2=9.7;
    arc+=k.segment([dx+Math.cos(a)*r1,dy+Math.sin(a)*r1,23.2],[dx+Math.cos(a)*r2,dy+Math.sin(a)*r2,23.2],P);
  }
  html+=L(arc,{tone:"lo"})+S(k.cylinder(dx,dy,2,23,1.2,P,20),{tone:"lo"});
  html+="<path data-part=\"budget-needle\" class=\"iso-line\" data-tone=\"hi\" data-free=\"end\" d=\""+k.lineOnTop([dx,dy],[dx-8,dy],24.5,P)+"\" style=\"stroke-width:1.2\"/>";
  html+=D([at(dx,dy,24.5)],{size:0.7,tone:"hi"});
  for(const x of [COMPOSER.x+15,COMPOSER.x+31]) html+=S(k.cylinder(x,COMPOSER.y+38,2.6,20,4,P,20),{tone:"lo"});
  html+=box(OUTPUT,8);
  const p={x:OUTPUT.x+7,y:OUTPUT.y+8,w:OUTPUT.w-14,d:OUTPUT.d-17,r:1};
  html+=ruledCard(p,8.5,5,"hi");
  for(const y of [OUTPUT.y+5,OUTPUT.y+OUTPUT.d-5])
    html+=S(k.slabOf({x:OUTPUT.x+3,y:y,w:OUTPUT.w-6,d:2,r:0.6},8,3,P,3,0.3),{tone:"lo"});
  return html;
}
function motionPath(id,points,z){return "<path id=\""+id+"\" d=\""+k.pathOf(points.map(p=>at(p[0],p[1],z)))+"\"/>";}
const defs="<defs>"+motionPath("bypass-motion",[[64,-77],[105,-77],[165,-77],[211,-47]],5)+motionPath("refine-motion",[[64,-35],[66,15],[92,53],[153,64],[179,10],[170,-94],[91,-111],[-42,-110],[-45,-55]],5)+motionPath("final-motion",[[-8,35],[40,80],[100,101],[203,101],[223,55]],8)+"</defs>";
const flows="<g class=\"route-bypass flow-token\"><circle r=\"2.4\"><animateMotion dur=\"5.4s\" repeatCount=\"indefinite\"><mpath href=\"#bypass-motion\"/></animateMotion></circle></g><g class=\"route-refine flow-token\"><circle r=\"2.4\"><animateMotion dur=\"8s\" repeatCount=\"indefinite\"><mpath href=\"#refine-motion\"/></animateMotion></circle></g><g class=\"route-final flow-token\"><circle r=\"2.4\"><animateMotion dur=\"6.2s\" repeatCount=\"indefinite\"><mpath href=\"#final-motion\"/></animateMotion></circle></g>";
const svg=k.figureSvg({width:W,height:H,label:"Bàn máy isometric mô tả pipeline CADRE hai lượt, cổng copy-protection, intent resolver, target API specification và adaptive prompt budget.",body:[defs,base(),frameTracks(),inputs(),engine(),gate(),intentAndSpec(),composerAndOutput(),flows]});
const figure=k.plateHtml({
  fig:"FIG 01",title:"CADRE Official · pipeline hai lượt",
  hint:"Kéo thanh context; thử các cổng bypass",
  readout:"Slack +472 · pass 2 cap 2304 · left-min 256",
  keys:[
    {mark:"lit",label:"Thẻ sáng: target API spec được phân giải từ repo pool"},
    {mark:"raised",label:"Cụm máy: các bước CADRE trên context và draft"},
    {mark:"flat",label:"Khay đầu vào: left context, retrieved chunks, pool files"},
    {mark:"line",label:"Đường chạy: bypass draft, refine có điều kiện, completion"}
  ],
  caption:"Bám theo cadre_official/cadre/pipeline.py và prompt_builder.py. Retriever, pool_files và generate_fn được truyền từ bên ngoài; hình chỉ mô phỏng luồng, không gọi model.",
  body:"<div class=\"iso-stage\" tabindex=\"0\" role=\"group\" aria-label=\"Mô hình tương tác kiến trúc CADRE\">"+svg+"</div>"
});
const partsHtml=Object.entries(PARTS).map(([name,items])=>"<section><h3>"+name+"</h3><ul>"+items.map(x=>"<li>"+esc(x)+"</li>").join("")+"</ul></section>").join("");
const controls=
"<section class=\"controls\" aria-label=\"Điều khiển mô phỏng\">"+
"<div class=\"control-card\"><div class=\"control-heading\"><div><span class=\"eyebrow\">BUDGET SLACK</span><h2>Độ dài left context</h2></div><output id=\"left-value\" for=\"left-tokens\">1000 tokens</output></div>"+
"<input id=\"left-tokens\" type=\"range\" min=\"0\" max=\"2600\" step=\"16\" value=\"1000\" aria-label=\"Số token left context\"/><div class=\"range-labels\"><span>0</span><span>ngắn</span><span>dài</span><span>2600</span></div>"+
"<p>Code tính: <code>slack = 3072 − (L + 1536 + 64)</code>. Nhánh đổi khi slack qua 0.</p></div>"+
"<div class=\"control-card\"><span class=\"eyebrow\">SAFETY GATES</span>"+
"<label class=\"switch-row\"><input id=\"copy-gate\" type=\"checkbox\"/><span class=\"switch\"></span><span><strong>Draft trùng dòng context</strong><small>Dòng không rỗng đầu similarity ≥ 0.95 → trả draft, bỏ refine.</small></span></label>"+
"<label class=\"switch-row\"><input id=\"spec-found\" type=\"checkbox\" checked/><span class=\"switch\"></span><span><strong>Tìm thấy target spec</strong><small>Không tìm thấy spec → trả draft, không gọi pass 2.</small></span></label>"+
"<label class=\"switch-row\"><input id=\"cssi-on\" type=\"checkbox\"/><span class=\"switch\"></span><span><strong>Chèn base CSSI</strong><small>Tùy chọn; mặc định <code>use_base_cssi=False</code>.</small></span></label></div></section>"+
"<section class=\"facts\">"+
"<article><span class=\"step\">01 / INPUT</span><h2>Context đến từ đâu?</h2><p>Active file mang <code>file_path</code> và <code>left_context</code>. Baseline retriever được gọi bên ngoài rồi truyền <code>retrieved_chunks</code> vào. <code>pool_files</code> được dùng riêng để tìm spec/CSSI.</p></article>"+
"<article><span class=\"step\">02 / DRAFT</span><h2>Pass 1 và cổng bảo vệ</h2><p>Generator tạo draft <code>z₀</code>. CADRE so dòng đầu tiên với tối đa 6 retrieved chunks và 80 dòng cuối của left context. Similarity ≥ 0.95 thì trả <code>z₀</code>; nếu không, parser trích receiver, callees và token stems.</p></article>"+
"<article><span class=\"step\">03 / SPEC</span><h2>Intent → target API</h2><p><code>TargetSpecResolver</code> dò khai báo trong repo pool bằng candidate tokens và giữ tên class trong signature. Nếu top-3 retrieved chunks đã có definition chính, hoặc không tìm thấy spec, pipeline trả draft mà không chạy pass 2.</p></article>"+
"<article><span class=\"step\">04 / REFINE</span><h2>Ghép prompt lượt hai</h2><p>Channel A coalesce raw chunks theo filepath; CSSI là tùy chọn; target spec thành banner; tiếp theo là file path và suffix left context. Cùng <code>generate_fn</code> được gọi lần nữa để sinh completion.</p></article></section>"+
"<section class=\"budget-note\"><div><span class=\"eyebrow\">WINDOW SLACK GUARD · DEFAULTS TRONG CODE</span><h2><code>W=3072</code> · <code>B_mid=1536</code> · <code>headroom=64</code></h2><p>Slack ≥ 0: pass 2 đặt cross-file cap 2304, min-left parameter 256. Slack &lt; 0: cap 1536, min-left parameter 1024. PromptBuilder còn khấu trừ path, CSSI/spec và headroom khi lắp prompt.</p></div><div class=\"budget-equation\"><code>slack = W − (L + B_mid + headroom)</code><span id=\"budget-state\">slack +472 → cap 2304 / left-min 256</span></div></section>"+
"<details class=\"parts\"><summary>"+partCount+" chi tiết được dựng trong Figure</summary><div>"+partsHtml+"</div></details>"+
"<footer><span>Source: <code>pipeline.py</code>, <code>intent_parser.py</code>, <code>spec_resolver.py</code>, <code>cssi.py</code>, <code>prompt_builder.py</code>.</span><button id=\"theme-toggle\" type=\"button\" aria-pressed=\"false\">Chuyển nền sáng/tối</button></footer>";

const live=
"const P="+JSON.stringify(P)+";"+
"const LEFT=document.querySelector('#left-tokens'),LV=document.querySelector('#left-value'),RO=document.querySelector('[data-readout]'),BS=document.querySelector('#budget-state');"+
"const COPY=document.querySelector('#copy-gate'),SPEC=document.querySelector('#spec-found'),CSSI=document.querySelector('#cssi-on');"+
"const NEEDLE=document.querySelector('[data-part=\"budget-needle\"]'),CHANNEL=document.querySelector('[data-part=\"cssi-channel\"]'),ARM=document.querySelector('[data-part=\"gate-arm\"]'),STAGE=document.querySelector('.iso-stage');"+
"const DIAL=["+(COMPOSER.x+72)+","+(COMPOSER.y+38)+"],DZ=24.5;"+
"function updateBudget(){const left=Number(LEFT.value),slack=3072-(left+1536+64),roomy=slack>=0,cap=roomy?2304:1536,min=roomy?256:1024;const msg='Slack '+(slack>=0?'+':'')+slack+' · pass 2 cap '+cap+' · left-min '+min;LV.value=left+' tokens';RO.textContent=msg;BS.textContent='slack '+(slack>=0?'+':'')+slack+' → cap '+cap+' / left-min '+min;const a=(-135+270*Math.max(0,Math.min(1,(left+100)/2700)))*Math.PI/180;NEEDLE.setAttribute('d',lineOnTop(DIAL,[DIAL[0]+Math.cos(a)*8.2,DIAL[1]+Math.sin(a)*8.2],DZ,P));STAGE.setAttribute('aria-valuetext',msg);}"+
"function updateRoute(){const bypass=COPY.checked||!SPEC.checked;document.querySelectorAll('.route-bypass').forEach(e=>e.classList.toggle('route-active',bypass));document.querySelectorAll('.route-refine,.route-final').forEach(e=>e.classList.toggle('route-muted',bypass));const pivot=iso([42,-14,22],P).join(' ');ARM.setAttribute('transform','rotate('+(bypass?34:-12)+' '+pivot+')');STAGE.setAttribute('aria-description',COPY.checked?'Copy gate: trả draft z₀.':(!SPEC.checked?'Không có spec: trả draft z₀.':'Có spec: tiếp tục prompt pass 2.'));}"+
"LEFT.addEventListener('input',updateBudget);COPY.addEventListener('change',updateRoute);SPEC.addEventListener('change',updateRoute);CSSI.addEventListener('change',()=>CHANNEL.classList.toggle('channel-on',CSSI.checked));"+
"document.querySelector('#theme-toggle').addEventListener('click',e=>{const light=document.body.dataset.theme!=='light';document.body.dataset.theme=light?'light':'dark';e.currentTarget.setAttribute('aria-pressed',String(light));});"+
"const motionSvg=STAGE.querySelector('svg'),reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');let onscreen=false;function syncMotion(){if(onscreen&&!reducedMotion.matches&&document.visibilityState==='visible')motionSvg.unpauseAnimations();else motionSvg.pauseAnimations();}new IntersectionObserver(entries=>{onscreen=entries.some(e=>e.isIntersecting);syncMotion();}).observe(STAGE);document.addEventListener('visibilitychange',syncMotion);reducedMotion.addEventListener('change',syncMotion);syncMotion();"+
"updateBudget();updateRoute();";

const style="<style>"+
".iso-page{width:min(1280px,100% - 32px);margin:38px auto 72px}.iso-plate{border-radius:20px;overflow:hidden}.iso-figure{margin:0}.iso-stage{outline:none;touch-action:pan-y}.iso-stage:focus-visible{outline:2px solid var(--anatomy-lit);outline-offset:4px}.iso-svg{aspect-ratio:1160/730}.iso-legend{display:grid;gap:13px;padding:16px 20px 20px}.iso-keys{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 22px}.route-muted{opacity:.15!important}.route-bypass{opacity:.3;transition:opacity .45s var(--anatomy-ease)}.route-bypass.route-active{opacity:1}.route-refine,.route-final{transition:opacity .45s var(--anatomy-ease)}.flow-token{fill:var(--anatomy-lit);opacity:.82;filter:drop-shadow(0 0 5px var(--anatomy-lit))}.flow-token.route-muted{opacity:0!important}[data-part=\"cssi-channel\"]{opacity:.36;transition:opacity .45s var(--anatomy-ease)}[data-part=\"cssi-channel\"].channel-on{opacity:1}"+
".controls{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0}.control-card,.facts article,.budget-note,.parts{border:1px solid color-mix(in srgb,var(--anatomy-text) 20%,transparent);border-radius:16px;background:var(--anatomy-card);padding:20px}.control-heading{display:flex;align-items:start;justify-content:space-between;gap:12px}.eyebrow,.step{font:600 10px/1.4 var(--anatomy-mono);letter-spacing:.12em;color:var(--anatomy-muted)}h2{font-size:17px;line-height:1.3;margin:5px 0 12px;color:var(--anatomy-strong)}.control-heading output{font:600 13px var(--anatomy-mono);color:var(--anatomy-strong);padding-top:19px;white-space:nowrap}"+
"input[type=range]{width:100%;accent-color:var(--anatomy-lit);margin:18px 0 4px}.range-labels{display:flex;justify-content:space-between;color:var(--anatomy-muted);font:11px var(--anatomy-mono)}.control-card p,.facts p,.budget-note p{font-size:13px;line-height:1.65;color:var(--anatomy-text);margin:12px 0 0}code{font:12px var(--anatomy-mono);color:var(--anatomy-strong)}"+
".switch-row{display:grid;grid-template-columns:0 38px 1fr;align-items:center;gap:12px;position:relative;margin-top:16px;cursor:pointer}.switch-row input{position:absolute;opacity:0;width:1px;height:1px}.switch{height:22px;width:38px;border-radius:12px;background:var(--anatomy-faint);border:1px solid color-mix(in srgb,var(--anatomy-text) 32%,transparent);transition:background .25s}.switch:before{content:'';display:block;width:14px;height:14px;margin:3px;border-radius:50%;background:var(--anatomy-text);transition:transform .25s var(--anatomy-ease),background .25s}.switch-row input:checked + .switch{background:var(--anatomy-lit)}.switch-row input:checked + .switch:before{transform:translateX(16px);background:var(--anatomy-paper)}.switch-row input:focus-visible + .switch{outline:2px solid var(--anatomy-lit);outline-offset:3px}.switch-row strong,.switch-row small{display:block}.switch-row strong{font-size:13px;color:var(--anatomy-strong)}.switch-row small{font-size:11px;line-height:1.5;color:var(--anatomy-text);margin-top:3px}"+
".facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0}.facts h2{margin:6px 0 4px}.facts p{margin:0}.budget-note{display:flex;justify-content:space-between;align-items:center;gap:24px;margin:16px 0}.budget-note h2{font-size:15px;margin:5px 0}.budget-note p{max-width:720px;margin:0}.budget-equation{display:grid;gap:8px;text-align:right;white-space:nowrap}.budget-equation span{font:12px var(--anatomy-mono);color:var(--anatomy-strong)}"+
".parts{margin-top:16px}.parts summary{cursor:pointer;color:var(--anatomy-strong);font-weight:600}.parts>div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:14px}.parts h3{font-size:12px;color:var(--anatomy-strong);margin:0 0 6px}.parts ul{padding-left:17px;margin:0;color:var(--anatomy-text);font-size:11px;line-height:1.7}footer{display:flex;justify-content:space-between;align-items:center;gap:18px;margin-top:24px;color:var(--anatomy-muted);font-size:11px}footer button{border:1px solid color-mix(in srgb,var(--anatomy-text) 28%,transparent);border-radius:999px;padding:9px 13px;background:var(--anatomy-card);color:var(--anatomy-text);cursor:pointer}"+
".page-head{margin:0 0 18px;padding:0 4px}.page-head h1{font-size:clamp(26px,4vw,42px);line-height:1.08;letter-spacing:-.04em;color:var(--anatomy-strong);margin:8px 0}.page-head p{max-width:750px;color:var(--anatomy-text);margin:0;font-size:14px}"+
"@media(max-width:760px){.iso-page{width:calc(100% - 20px);margin:18px auto 40px}.controls,.facts{grid-template-columns:1fr}.iso-keys{grid-template-columns:1fr}.budget-note{align-items:flex-start;flex-direction:column}.budget-equation{text-align:left;white-space:normal}.parts>div{grid-template-columns:1fr 1fr}footer{align-items:flex-start;flex-direction:column}.iso-plate-corners{font-size:10px}}@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}"+
"</style>";
const body="<header class=\"page-head\"><span class=\"eyebrow\">ARCHITECTURE ANATOMY / SOURCE-ALIGNED</span><h1>CADRE Official nhìn như một cỗ máy</h1><p>Một lượt draft, một cổng dừng an toàn và một lượt refine có điều kiện — dựng từ mã nguồn hiện tại.</p></header>"+figure+controls;
const html=k.pageHtml({title:"CADRE Official · kiến trúc pipeline hai lượt",theme:LIGHT?"light":"dark",body:body,script:kitScript()+live,width:1280}).replace("</head>",style+"</head>").replace('<html lang="en">','<html lang="vi">');
mkdirSync(HERE,{recursive:true});
const output=join(HERE,LIGHT?"cadre_official_architecture-light.html":"index.html");
writeFileSync(output,html);
const standaloneSvg=k.figureSvg({width:W,height:H,label:"CADRE Official · sơ đồ kiến trúc isometric",body:[defs,base(),frameTracks(),inputs(),engine(),gate(),intentAndSpec(),composerAndOutput(),flows],standalone:LIGHT?"light":"dark"});
writeFileSync(join(HERE,LIGHT?"cadre_official_architecture-light.svg":"diagram.svg"),standaloneSvg.replace("<svg ","<svg width=\""+W+"\" height=\""+H+"\" "));
console.log("Wrote "+output+"; "+partCount+" grouped physical details; theme="+(LIGHT?"light":"dark"));

