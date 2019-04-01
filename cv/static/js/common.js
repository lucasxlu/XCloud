$('.btn').hover(function(){
    $(this).find('i').addClass('animated wobble');
    setTimeout(function(){
        $(this).find('i').removeClass('animated wobble');
    }, 1000);
},function(){
    $(this).find('i').removeClass('animated wobble');
});
$(document).ready(function() {
    //首先将#amz-go-top隐藏
    $("#amz-go-top").hide();
    //当滚动条的位置处于距顶部100像素以下时，跳转链接出现，否则消失
    $(function() {
        $(window).scroll(function() {
            if ($(window).scrollTop() > 100) {
                $("#amz-go-top").fadeIn(1500);
            } else {
                $("#amz-go-top").fadeOut(1500);
            }
        });
        //当点击跳转链接后，回到页面顶部位置
        $("#amz-go-top").click(function() {
            $('body,html').animate({
                    scrollTop: 0
                },
                500);
            return false;
        });
    });
});