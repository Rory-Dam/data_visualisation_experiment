(this.webpackJsonpstreamlit_viewership_component=this.webpackJsonpstreamlit_viewership_component||[]).push([[0],{126:function(t,e,a){},128:function(t,e,a){"use strict";a.r(e);var n,i=a(12),r=a.n(i),o=a(45),s=a.n(o),c=a(9),l=a(0),p=a(1),u=a(2),d=a(3),h=a(21),m=a(8),f=a(47),v=(a(126),a(13)),y=function(t){Object(u.a)(a,t);var e=Object(d.a)(a);function a(){var t;Object(l.a)(this,a);for(var i=arguments.length,r=new Array(i),o=0;o<i;o++)r[o]=arguments[o];return(t=e.call.apply(e,[this].concat(r))).componentDidMount=function(){var e=t.get_dimensions(),a=Object(c.a)(e,3),i=a[0],r=a[1],o=a[2];var s=m.i("#"+n).append("svg").attr("width",i+o.left+o.right).attr("height",r+o.top+o.bottom).call((function(t){var e=m.i(t.node().parentNode),a=parseInt(t.style("width")),n=parseInt(t.style("height")),i=a/n;function r(){var a=parseInt(e.style("width"));t.attr("width",a),t.attr("height",Math.round(a/i)),h.a.setFrameHeight()}t.attr("viewBox","0 0 "+a+" "+n).attr("perserveAspectRatio","xMinYMid").call(r),m.i(window).on("resize."+e.attr("id"),r)})).append("g").attr("transform","translate(".concat(o.left,", ").concat(o.top,")"));s.append("defs").append("svg:clipPath").attr("id","clip").append("svg:rect").attr("width",i).attr("height",r).attr("x",0).attr("y",0);s.append("rect").attr("id","zoomRect").attr("width",i+o.left+o.right).attr("height",r).style("fill","none").style("pointer-events","all").attr("transform","translate("+-o.left+",0)"),s.append("g").attr("transform","translate(0,".concat(r,")")).attr("id","myXaxis"),s.append("g").attr("id","myYaxis"),m.i("body").append("div").attr("class","svg-tooltip").style("font-family",'-apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple   Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"').style("border-radius",".1rem").style("font-size","14px").style("max-width","320px").style("text-overflow","ellipsis").style("white-space","pre").style("z-index","300").style("background","rgba(69,77,93,.9)").style("color","#fff").style("display","block").style("padding",".2rem .4rem").style("position","absolute").style("visibility","hidden"),s.append("text").attr("id","yLabel").attr("transform","rotate(-90)").attr("y",-o.left).attr("x",-r/2).attr("dy","1em").style("font-size",20).style("text-anchor","middle"),s.append("g").attr("class","legendSize").attr("transform","translate(".concat(i,", 20)")),h.a.setFrameHeight()},t.render=function(){return n="bubble_component_"+t.props.args.key,Object(v.jsxs)("span",{children:[Object(v.jsx)("div",{id:n}),Object(v.jsx)("div",{className:"button_container",id:"chapter_button_container"}),Object(v.jsxs)("div",{className:"button_container",children:[Object(v.jsx)("a",{id:"show_all_button",className:"ripple_button",children:"Show All"}),Object(v.jsx)("a",{id:"swap_view_button",className:"ripple_button",children:"Swap View"})]})]})},t}return Object(p.a)(a,[{key:"get_data",value:function(){for(var t=this.props.args.data,e=[],a=t.columns,n=function(n){var i=m.f(0,a).map((function(e){var a=t.getCell(n,e),i=(a.classNames,a.content);a.id,a.type;return null===i||void 0===i?void 0:i.toString()}));e.push({index:n,episode:parseInt(i[1]),act:i[2],chapter:i[3],segment:i[4],start_time:parseFloat(i[5]),end_time:parseFloat(i[6]),viewership:parseFloat(i[7])})},i=1;i<t.rows;i++)n(i);return e}},{key:"get_metadata",value:function(){return this.props.args.metadata}},{key:"get_theme",value:function(){return this.props.theme}},{key:"get_dimensions",value:function(){var t={top:80,right:150,bottom:80,left:80};return[1200-t.left-t.right,600-t.top-t.bottom,t]}},{key:"componentDidUpdate",value:function(){var t,e,a=this,i=this.get_data(),r=this.get_metadata(),o=this.get_dimensions(),s=Object(c.a)(o,3),l=s[0],p=s[1],u=(s[2],r.colour),d=m.i("#"+n).select("svg").select("g"),v={};r.chapter_order.map((function(t,e){t in v||(v[t]=e+1)}));var y=Object.keys(v).length,g=i.sort((function(t,e){return t.end_time-t.start_time>e.end_time-e.start_time?-1:t.end_time-t.start_time<e.end_time-e.start_time?1:0})),b=m.g().domain([0,y+1]).range([0,l]),w=m.g().domain([0,1.1*m.c(i,(function(t){return t.viewership}))]).range([p,0]),_=m.g().domain([0,Math.sqrt(m.d(i,(function(t){return t.end_time-t.start_time})))/Math.PI]).range([0,l/m.c(i,(function(t){return t.episode}))/2*.5]),x=m.i("#myXaxis"),k=m.i("#myYaxis"),j={};Object.keys(v).map((function(t,e){j[e+1]=t})),x.call(m.a(b).ticks(y).tickFormat((function(t){return j[t]}))),k.call(m.b(w).tickSize(-l)),k.selectAll("text").style("font-size","15");var A=m.i("body").select(".svg-tooltip"),O=!1;d.select("#dataCircle").empty()||"episode"===d.select("#dataCircle").attr("view")&&(O=!0);var S=i.map((function(t){return t.act})).filter((function(t,e,a){return a.indexOf(t)===e})),z=Object.fromEntries(S.map((function(t){return[t,0]})));i.map((function(t){t.episode>z[t.act]&&(z[t.act]=t.episode)}));var M=Object.entries(z).map((function(t){return{act:t[0],x:t[1]+.5}}));console.log(M),d.selectAll(".actLine").data(M).join("path").attr("clip-path","url(#clip)").attr("d",(function(){var t=m.e();return t.moveTo(0,0),t.lineTo(0,p),t})).attr("fill","none").attr("stroke","black").attr("stroke-width",1).style("stroke-dasharray","4, 4").attr("class","actLine"),d.selectAll(".actLabel").data(M).join("text").text((function(t){return t.act})).attr("transform","translate(0, -5)").style("opacity","0").style("text-anchor","end").attr("class","actLabel"),d.selectAll("#dataCircle").data(g).join("circle").attr("id","dataCircle").attr("view","chapter").attr("clip-path","url(#clip)").attr("r",(function(t){return _(Math.sqrt(t.end_time-t.start_time)/Math.PI)})).attr("cx",(function(t){return b(v[t.chapter])})).attr("cy",(function(t){return w(t.viewership)})).attr("fill",(function(t){return u[t.chapter]})).style("opacity","0.7").style("pointer-events","auto").on("mouseover",(function(t,e){m.i(t.target).attr("stroke-width","1").attr("stroke","black"),A.style("visibility","visible").text("Viewership: ".concat(Math.round(e.viewership),"\nEpisode: ").concat(e.episode,"\n")+"Act: ".concat(e.act,"\nChapter: ").concat(e.chapter,"\nSegment: ").concat(e.segment,"\n")+"Start(s): ".concat(e.start_time,"\nEnd(s): ").concat(e.end_time))})).on("mousemove",(function(t){A.style("top",t.pageY-10+"px").style("left",t.pageX+10+"px")})).on("mouseout",(function(t){m.i(t.target).attr("stroke-width",null).attr("stroke",null),A.style("visibility","hidden")})).attr("view","chapter"),d.select("#yLabel").text("".concat(r.metric));var C=Math.pow(10,Math.floor(Math.log10(m.c(i,(function(t){return t.end_time-t.start_time}))))),F=[.1,.25,.5,1].map((function(t){return t*C})),I=F.map((function(t){return _(Math.sqrt(t)/Math.PI)})),E=m.h().domain(F.map((function(t){return"".concat(t,"s")}))).range(I),L=f.a.legendSize().ascending(!0).title("Segement Time").shape("circle").shapePadding(5).labelOffset(10).orient("vertical").scale(E);d.select(".legendSize").call(L).selectAll("circle").style("opacity",.5),d.select(".legendSize").selectAll("text").style("font-size","20");var T=m.j().scaleExtent([1,20]).extent([[0,0],[l,p]]).on("zoom",(function(t){if(!isNaN(t.transform.k)){var e=t.transform.rescaleY(w);k.call(m.b(e).tickSize(-l)),k.selectAll("text").style("font-size","15"),d.selectAll("#dataCircle").attr("cy",(function(t){return e(t.viewership)}))}}));function N(t){var e=t.clientX-t.target.offsetLeft,a=t.clientY-t.target.offsetTop,n=document.createElement("div");n.style.left=e+"px",n.style.top=a+"px",t.target.append(n),setTimeout((function(){n.remove()}),800)}d.select("#zoomRect").call(T),null===(t=document.getElementById("zoomRect"))||void 0===t||t.dispatchEvent(new WheelEvent("wheel",{deltaY:1})),null===(e=document.getElementById("zoomRect"))||void 0===e||e.dispatchEvent(new WheelEvent("wheel",{deltaY:-1})),O&&this.swapView(r.chapter_order,0),m.i("#swap_view_button").on("click",(function(t){N(t),a.swapView(r.chapter_order)})),m.i("#show_all_button").on("click",(function(t){N(t),a.showAll()}));var Y=m.i("#chapter_button_container"),X=this.swapShowChapter;Y.selectAll(".ripple_button").data(Object.keys(v)).join("a").attr("id",(function(t){return"show_".concat(t,"_button")})).attr("class","ripple_button").style("background-color",(function(t){return u[t]})).style("color",(function(t){return function(t){t=t.replace("#","");var e=(.299*parseInt(t.substr(0,2),16)+.587*parseInt(t.substr(2,2),16)+.114*parseInt(t.substr(4,2),16))/255;return(Math.max(e,.1)+.05)/(Math.min(e,.1)+.05)>=4.5?"#000000":"#FFFFFF"}(u[t])})).text((function(t){return t})).on("click",(function(t,e){m.i(this).style("opacity","1"===m.i(this).style("opacity")?"0.5":"1"),X(e)})),h.a.setFrameHeight()}},{key:"swapView",value:function(t){var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:800,a=m.i("#"+n).select("svg").select("g");"chapter"!==a.select("#dataCircle").attr("view")?"episode"!==a.select("#dataCircle").attr("view")||function(t){var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:800,a={top:80,right:150,bottom:80,left:80},i=1200-a.left-a.right,r=600-a.top-a.bottom,o=m.i("#"+n).select("svg").select("g"),s={};null===t||void 0===t||t.map((function(t,e){t in s||(s[t]=e+1)}));var c=Object.keys(s).length,l=m.g().domain([0,c+1]).range([0,i]),p=m.i("#myXaxis"),u={};Object.keys(s).map((function(t,e){u[e+1]=t})),p.transition().duration(e).call(m.a(l).ticks(c).tickFormat((function(t){return u[t]}))),o.selectAll("#dataCircle").attr("view","chapter").transition().duration(e).attr("cx",(function(t){return l(s[t.chapter])})),o.selectAll(".actLine").transition().duration(e).attr("d",(function(t){var e=m.e();return e.moveTo(0,0),e.lineTo(0,r),e})),o.selectAll(".actLabel").transition().duration(e).style("opacity","0").attr("transform","translate(0, -5)")}(t,e):function(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:800,e={top:80,right:150,bottom:80,left:80},a=1200-e.left-e.right,i=600-e.top-e.bottom,r=m.i("#"+n).select("svg").select("g"),o=0;r.selectAll("#dataCircle").each((function(t){t.episode>o&&(o=t.episode)}));var s=m.g().domain([0,o+1]).range([0,a]);m.i("#myXaxis").transition().duration(t).call(m.a(s).ticks(o).tickFormat((function(t){if(t>0&&t<=o)return"Episode ".concat(t)}))),r.selectAll("#dataCircle").attr("view","episode").transition().duration(t).attr("cx",(function(t){return s(t.episode)})),r.selectAll(".actLine").transition().duration(t).attr("d",(function(t){console.log(t);var e=m.e();return e.moveTo(s(t.x),0),e.lineTo(s(t.x),i),e})),r.selectAll(".actLabel").transition().duration(t).style("opacity","100").attr("transform",(function(t){return"translate(".concat(s(t.x),", -5)")}))}(e)}},{key:"showAll",value:function(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:400;m.i("#"+n).select("svg").select("g").selectAll("#dataCircle").transition().duration(t).style("opacity","0.7").style("pointer-events","auto"),m.i("#chapter_button_container").selectAll(".ripple_button").style("opacity","1")}},{key:"swapShowChapter",value:function(t){var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:400;m.i("#"+n).select("svg").select("g").selectAll("#dataCircle").transition().duration(e).style("opacity",(function(e){return e.chapter===t?"0.7"===m.i(this).style("opacity")?"0":"0.7":m.i(this).style("opacity")})).style("pointer-events",(function(e){return e.chapter===t?"auto"===m.i(this).style("pointer-events")?"none":"auto":m.i(this).style("pointer-events")}))}}]),a}(h.b),g=Object(h.c)(y);s.a.render(Object(v.jsx)(r.a.StrictMode,{children:Object(v.jsx)(g,{})}),document.getElementById("root"))}},[[128,1,2]]]);
//# sourceMappingURL=main.9683ae47.chunk.js.map