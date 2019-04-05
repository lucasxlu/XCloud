/**
 * Created by 29140 on 2016/11/1.
 */
var nav = $("#navbar").width() - 65 - 284;//宽度navbar-brand：60,form:284(此处为估算值)
var linum = $(".navbar-nav li").length;
var licon = $(".navbar-nav").width();
$(function () {
    countliw(nav, linum, licon);
})
$(window).resize(function () {//浏览器窗口发生变化时的监听事件
    checknav($(window).width());
});

function checknav(a) {
    //32为navbar左右内边距和；
    if (a <= 992) {
        $("#fsearch").hide();
        countliw(a - 65 - 10 - 32, linum, licon);
        if (a <= 750) {
            $("#fsearch").show();
        }
    } else {
        countliw(a - 65 - 284 - 32, linum, licon);
        $("#fsearch").show();
    }
}

function countliw(a, b, c) {
    var spac = (a - c) / b;
    $("#navbar li").each(function (index) {
        $(this).css({"paddingRight": spac + "px"});
    })
}

function search() {
    newurl = 'https://www.baidu.com/s?wd=' + $('#searchContent').val().trim();
    console.log(newurl);
    window.location.assign(newurl);
}

/*
$('.carousel').carousel()*/
