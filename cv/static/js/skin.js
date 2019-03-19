var newUrl_base64;

function getUrlBase64(img) {
    function getBase64Image(img, width, height) { //width、height调用时传入具体像素值，控制大小 ,不传则默认图像大小
        var canvas = document.createElement("canvas");
        canvas.width = width ? width : img.width;
        canvas.height = height ? height : img.height;

        var ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        var dataURL = canvas.toDataURL();
        return dataURL;
    }

    var image = new Image();
    image.crossOrigin = 'anonymous';
    image.src = img;
    var deferred = $.Deferred();
    if (img) {
        image.onload = function () {
            deferred.resolve(getBase64Image(image)); //将base64传给done上传处理
        }
        return deferred.promise(); //问题要让onload完成后再return sessionStorage['imgTest']
    }
}


/*页面初始化及一些点击事件+移动端首页点击事件*/
window.onload = function () {
    /*复选框最多可选3个start*/
    $('input[type=checkbox]').click(function () {//checkbox的input标签点击事件
        $("input[name='apk[]']").attr('disabled', true);//设置name为apk[]的input标签的属性为true,表示不可选
        if ($("input[name='apk[]']:checked").length >= 3) {//选中个数大于3
            $("input[name='apk[]']:checked").attr('disabled', false);
        } else {
            $("input[name='apk[]']").attr('disabled', false);
        }
    });
    /*复选框最多可选3个end*/


    //页面初始化
    $(".secondary_nav_box ").eq(0).click();
    //跑马灯
    // $('.str5').liMarquee({
    //     hoverstop: false //鼠标停留不停止
    // });
    $(window).mouseup(function () {
        document.onmousedown = null;
        document.onmousemove = null;
    });

    /*图像标签URL点击*/
    $(".picture .immigration").on("click", function () {
        $(".picture .immigration_box").css("display", "block");
        $(".picture .immigration").css("display", "none");
    });
    /*低光照点击*/
    $(".the_immigration").on("click", function () {
        $(".the_immigration_box").css("display", "block");
        $(".the_immigration").css("display", "none");
    });
    /*图像去雾点击*/
    $(".fog_immigration").on("click", function () {
        $(".fog_immigration_box").css("display", "block");
        $(".fog_immigration").css("display", "none");
    });
    /*图像矫正*/
    $(".rectify_immigration").on("click", function () {
        $(".rectify_immigration_box").css("display", "block");
        $(".rectify_immigration").css("display", "none");
    });

    //移动视频播放
    $(".maddImgV").on("click", function () {
        var videow = $('.mvideo-s').width();
        var videoh = parseInt(videow * 0.55) + 'px';
        $(".maddImgV").css("display", "none");
        $(".mvideo-s").css("display", "block");

        var myVideo = $(".mvideo-s")[0];
        if (myVideo.paused) {
            $('.mvideo-s').css('height', '152px');
            myVideo.play();
        } else {

            myVideo.pause();
        }
    });
    //移动端点击
    $('.in3box').on('click', function () {
        var f = $(this).next('.in3bbox');
        if (f.is(':hidden')) {
            f.slideDown("fast");
            // $(this).css('background', '#2F94FF');
            // $(this).find('p').css('color', '#FFFFFF');
            $(this).children('.in3boximg2').attr('src', 'https://res-img2.huaweicloud.com/content/dam/cloudbu-site/archive/china/zh-cn/ei/experiencespace/v1/images/mobile/ei-arr-top@2x.png');
            $(this).css('border-bottom', 'none');

        } else {
            f.slideUp("fast");
            // $(this).css('background', '#FFFFFF');
            $(this).removeAttr("style");
            $(this).find('p').eq(0).css('color', '#383E63');
            // $(this).find('p').eq(1).css('color', '#666A75');
            $(this).children('.in3boximg2').attr('src', 'https://res-img2.huaweicloud.com/content/dam/cloudbu-site/archive/china/zh-cn/ei/experiencespace/v1/images/mobile/ei-arr-bottom@2x.png');
        }


    })


};
$(document).ready(function () {

});

/*================按钮拖拽==============*/
function btnMove(obj) {
    $("#word1,#word2,#word3").css('display', 'none');
    var wind = $(window).width();
    var box_width = $(".the_fyImage").eq(0).width()
    var list_wind = (wind - box_width) / 2;

    var e = window.event || arguments[0];
    //console.log("鼠标按下");
    document.onmousemove = function () {
        var mouseMoveE = document.event || arguments[0];
        //console.log("鼠标移动");
        obj.style.left = (mouseMoveE.clientX - e.offsetX - list_wind + 15.5) + "px";//clientX:页面X轴的位置 offsetX:相对于事件源的X坐标

        var imgDivWidth = (mouseMoveE.clientX - list_wind - e.offsetX + 15.5);
        divModified.style.width = imgDivWidth + "px";
        /*setWorDisplay(divModified);*/

        //按钮不能超出图片背景的范围
        if (imgDivWidth < 15.5) {
            divModified.style.width = "15.5px";
            obj.style.left = '15.5px';
        } else if (imgDivWidth > box_width - 15.5) {
            divModified.style.width = box_width + "px";
            obj.style.left = (box_width - 15.5) + "px";
        }
    }
}

function deleteEvent() {
    document.onmousedown = null;
    document.onmousemove = null;
}

//图像标签下面点击4张图片切换渲染
function shibieBtn(d, e) {
    $(".picture .immigration_box").css("display", "none");
    $(".picture .immigration").css("display", "block");
    $(".picture_vertifyimage").removeClass("picture_vertifyimage_active")
    $(e).addClass("picture_vertifyimage_active");

    $(".imgBox_picture").attr('src', $(e).find(".picture_min").attr("src"));
    $('.picture .loading_Box').css('display', 'block');
    $(".loading_list").css('display', 'block');
    $(".picture_min").removeClass("picture_frame")
    $(e).find(".picture_min").addClass("picture_frame");
    var picture = $("#identify_result_box");
    var picture1 = $("#identify_result_box1");
    var picture2 = $("#identify_result_box2");
    var picture3 = $("#identify_result_box3");


    if (d == 1) {
        $(".picture_max").css("width", "100%");
        $(".picture_max").css("height", "100%");
        $(".identify_result_box1").css("display", "none");
        $(".identify_result_box2").css("display", "none");
        $(".identify_result_box3").css("display", "none");
        $(".identify_result_box4").css("display", "none");
        $(".identify_result_box5").css("display", "none");
        $(".identify_result_box").css("display", "block");
        $("#identify_result_box").html(" ");
        $(".loading_list").css('display', 'block');
        $('.picture .loading_Box').css('display', 'block');
        setTimeout(function () {
            var html = "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Acne_Keloidalis_Nuchae</div><span>0.9636</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Infantile_Atopic_Dermatitis</div><span>0.0119</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Perioral_Dermatitis</div><span>0.0084</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Keratosis_Pilaris</div><span>0.0061</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Allergic_Contact_Dermatitis</div><span>0.0014</span></div></div>";

            $("#identify_result_box").html(html);
            $(".picture .loading_Box").css('display', 'none');
            $(".loading_list").css('display', 'none');
        }, 2000)
    }
    if (d == 2) {
        $(".picture_max").css("width", "100%");
        $(".picture_max").css("height", "100%");
        $(".identify_result_box").css("display", "none");
        $(".identify_result_box2").css("display", "none");
        $(".identify_result_box3").css("display", "none");
        $(".identify_result_box4").css("display", "none");
        $(".identify_result_box5").css("display", "none");
        $(".identify_result_box1").css("display", "block");
        $(".loading_list").css('display', 'block');
        $('.picture .loading_Box').css('display', 'block');
        picture1.html(" ");
        setTimeout(function () {
            var result = "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Bowen's_Disease</div><span>0.423</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Benign_Keratosis</div><span>0.1059</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Impetigo</div><span>0.0824</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Dermatofibroma</div><span>0.0777</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Nummular_Eczema</div><span>0.0638</span></div></div>";
            picture1.html(result);
            $(".picture .loading_Box").css('display', 'none');
            $(".loading_list").css('display', 'none');
        }, 2000)
    }
    if (d == 3) {
        $(".picture_max").css("width", "100%");
        $(".picture_max").css("height", "100%");
        $(".identify_result_box").css("display", "none");
        $(".identify_result_box1").css("display", "none");
        $(".identify_result_box3").css("display", "none");
        $(".identify_result_box4").css("display", "none");
        $(".identify_result_box5").css("display", "none");
        $(".identify_result_box2").css("display", "block");
        $('.picture .loading_Box').css('display', 'block');
        $(".loading_list").css('display', 'block');
        picture2.html(" ");
        setTimeout(function () {
            var html = "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Alopecia_Areata</div><span>0.993</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Hypertrichosis</div><span>0.0018</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Androgenetic_Alopecia</div><span>0.0015</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Acne_Keloidalis_Nuchae</div><span>0.001</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Follicular_Mucinosis</div><span>0.0007</span></div></div>";
            picture2.html(html);
            $(".picture .loading_Box").css('display', 'none');
            $(".loading_list").css('display', 'none');
        }, 2000)
    }
    if (d == 4) {
        $(".picture_max").css("width", "100%");
        $(".picture_max").css("height", "100%");
        $(".identify_result_box").css("display", "none");
        $(".identify_result_box1").css("display", "none");
        $(".identify_result_box2").css("display", "none");
        $(".identify_result_box4").css("display", "none");
        $(".identify_result_box5").css("display", "none");
        $(".identify_result_box3").css("display", "block");
        $('.picture .loading_Box').css('display', 'block');
        picture3.html("");
        $(".loading_list").css('display', 'block');
        setTimeout(function () {
            var html = "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Basal_Cell_Carcinoma</div><span>0.9161</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Bowen's_Disease</div><span>0.0229</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Actinic_solar_Damage(Actinic_Keratosis)</div><span>0.0132</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Disseminated_Actinic_Porokeratosis</div><span>0.0115</span></div></div>" +
                "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>Dilated_Pore_of_Winer</div><span>0.0103</span></div></div>";
            picture3.html(html);
            $(".picture .loading_Box").css('display', 'none');
            $(".loading_list").css('display', 'none');
        }, 2000)
    }
}

//图像标签上传图片
function img_upload() {
    var eleFile = document.querySelector('.local_upload_file');
    eleFile.addEventListener('change', function () {
        $('.loading_Box').css('display', 'block');
        $(".loading_list").css('display', 'block');
        var file = this.files[0];
        // 确认选择的文件是图片
        if (file.type.indexOf("image") == 0) {
            var reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = function (e) {
                // 图片base64化
                var newUrl = this.result;

                setTimeout(function () {
                    imageSize();
                }, 10)
                $("#identify_result_box5").html(" ");
                $('.loading_Box').css('display', 'block');
                $(".identify_result_box1").css("display", "none");
                $(".identify_result_box2").css("display", "none");
                $(".identify_result_box3").css("display", "none");
                $(".identify_result_box4").css("display", "none");
                $(".identify_result_box").css("display", "none");
                $(".identify_result_box5").css("display", "block");
                $(".imgBox_picture").attr('src', newUrl);
                $(".picture_vertifyimage").removeClass("picture_vertifyimage_active");
                $(".picture_min").removeClass("picture_frame");
                $("#identify_result_box5").html(" ");
                $(".dispose_btn").text("Processing");
                $(".dispose_btn").css("border-left", "1px solid #D4D4D4");
                $(".dispose_border").css("border", " 1px solid #D4D4D4");
                $(".prevent").css("display", "bock");
                newUrl_base64 = newUrl.replace(/^(data:\s*image\/(\w+);base64,)/g, '');
                $.ajax({
                    type: 'post',
                    // url: "https://wx.issmart.com.cn/web/test/lh/tp/index.php/api_pc/image/tag_image",
                    url: 'mcloud/skin',
                    data: {
                        "image": newUrl_base64
                    },
                    dataType: "json",
                    success: function (data) {
                        console.log(data);
                        if (data.status === 0) {
                            var dataList = data.results;

                            var html = '';
                            for (var i = 0; i < dataList.length; i++) {
                                var j = "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>" + dataList[i].disease + " </div><span>" + dataList[i].probability.toFixed(1) + "%</span></div></div>"
                                html += j
                            }
                            $("#identify_result_box5").html(html)
                            $(".loading_Box").css('display', 'none');
                            $(".loading_list").css('display', 'none');
                            $(".dispose_btn").text("Process");
                            $(".dispose_btn").css("border-left", "1px solid #2E97FF");
                            $(".dispose_border").css("border", " 1px solid #2E97FF");
                            $(".prevent").css("display", "none");
                            $("#local_upload").html("");
                            $("#local_upload").append("Upload <input type='file' maxlength='0' onclick='img_upload(this)' class='local_upload_file'>");
                        } else {
                            $("#local_upload").html("Invalid Image.");
                        }
                    },
                    error: function () {
                        $(".dispose_btn").text("Process");
                        $(".dispose_btn").css("border-left", "1px solid #2E97FF");
                        $(".dispose_border").css("border", " 1px solid #2E97FF");
                        $(".prevent").css("display", "none");
                    }
                })
            };
        }
    });
    $(".dispose_btn").text("Process");
    $(".dispose_btn").css("border-left", "1px solid #2E97FF");
    $(".dispose_border").css("border", " 1px solid #2E97FF");
    $(".prevent").css("display", "none");

}

//图像标签URL上传
function dispose_btn() {
    $("#identify_result_box5").html(" ");
    $(".identify_result_box1").css("display", "none");
    $(".identify_result_box2").css("display", "none");
    $(".identify_result_box3").css("display", "none");
    $(".identify_result_box4").css("display", "none");
    $(".identify_result_box").css("display", "none");
    $(".identify_result_box5").css("display", "block");
    var get_url = $.trim($(".dispose_url").val());
    //console.log(get_url)

    if (get_url == "") {
        close_succeed();
        $(".con_succeed_test_Login").text("Got it");
        $(".con_desc_symbol_succeed").text("Invalid image"); //弹窗
        $('.loading_Box').css('display', 'none');
        $(".loading_list").css('display', 'none');
    } else {
        $(".imgBox_picture").attr('src', get_url);
        $(".picture_vertifyimage").removeClass("picture_vertifyimage_active");
        $(".picture_min").removeClass("picture_frame");
        $('.loading_Box').css('display', 'block');
        $(".loading_list").css('display', 'block');
        setTimeout(function () {
            imageSize();
        }, 300)
        $(".dispose_btn").css("border-left", "1px solid #D4D4D4");
        $(".dispose_border").css("border", " 1px solid #D4D4D4");
        $(".dispose_btn").text("Processing");
        $(".prevent").css("display", "bock");
        $.ajax({
            type: 'post',
            url: "https://wx.issmart.com.cn/web/test/lh/tp/index.php/api_pc/image/tag_image",
            data: {
                "url": get_url
            },
            dataType: "json",
            success: function (data) {
                //console.log(data)
                if (data.state == "1") {
                    dataList = data.data.data;
                    var html = '';
                    for (var i = 0; i < dataList.length; i++) {
                        var j = "<div class='identify_result_ciy md-4'><div class='ciy_box'><div class='ciy_box_desc'>" + dataList[i].tag + "</div><span>" + dataList[i].confidence.toFixed(1) + "%</span></div></div>"
                        html += j
                    }
                    $('.loading_Box').css('display', 'none');
                    $(".loading_list").css('display', 'none');
                    $(".dispose_btn").text("Process");
                    $(".dispose_btn").css("border-left", "1px solid #2E97FF");
                    $(".dispose_border").css("border", " 1px solid #2E97FF");
                    $(".prevent").css("display", "none");
                    $("#identify_result_box5").html(html);
                } else {
                    close_succeed()
                    $(".con_succeed_test_Login").text("Got it");
                    $(".con_desc_symbol_succeed").text(data.msg); //弹窗
                    $(".dispose_btn").text("Process");
                    $(".dispose_btn").css("border-left", "1px solid #2E97FF");
                    $(".dispose_border").css("border", " 1px solid #2E97FF");
                    $('.loading_Box').css('display', 'none');
                    $(".loading_list").css('display', 'none');
                    $(".prevent").css("display", "none");
                    return false;
                }
            },
            error: function () {
                $(".dispose_btn").text("Process");
                $(".dispose_btn").css("border-left", "1px solid #2E97FF");
                $(".dispose_border").css("border", " 1px solid #2E97FF");
                $(".prevent").css("display", "none");
                $('.loading_Box').css('display', 'none');
                $(".loading_list").css('display', 'none');
                close_succeed()
                $(".con_succeed_test_Login").text("Got it");
                $(".con_desc_symbol_succeed").text(data.msg); //弹窗
            }
        })
    }
}


//图像标签图片自适应
function imageSize() {
    // 图片自适应
    var imgBox = $('.imgBox')[0];
    var img = imgBox.getElementsByTagName('img')[0];
    var imgBoxW = imgBox.offsetWidth;
    var imgBoxH = imgBox.offsetHeight;
    var $width = img.naturalWidth;//获取图片真实宽度
    var $height = img.naturalHeight;
    ratio = $width / $height; //图片的真实宽高比例
    if ($width > $height) {
        if (imgBoxW / ratio > imgBoxH) {
            var imgheight = imgBoxH;
            var ratio2 = $height / imgBoxH;
            var imgwidth = $width / ratio2;
            // imgratio = ratio2;
        } else {
            var imgwidth = imgBoxW;
            var imgheight = imgBoxW / ratio;
            // imgratio = $width / imgBoxW;
        }
        imgtop = (imgBoxH - imgheight) / 2;
        imgleft = (imgBoxW - imgwidth) / 2;

        img.style.width = imgwidth - 28 + 'px';
        img.style.height = imgheight - 28 + 'px';

    } else {
        if (imgBoxH * ratio > imgBoxW) {
            var imgwidth = imgBoxW;
            var ratio2 = $width / imgwidth;
            var imgheight = $height / ratio2;
            imgratio = ratio2;
        } else {
            var imgheight = imgBoxH;
            var imgwidth = imgBoxH * ratio;
            // imgratio = $height /imgheight;
        }
        imgtop = (imgBoxH - imgheight) / 2;
        imgleft = (imgBoxW - imgwidth) / 2;
        img.style.width = imgwidth - 28 + 'px';
        img.style.height = imgheight - 28 + 'px';
    }


}

//弹窗start
function showCon() {
    $(".cov").show();
}

function closeCon() {
    $(".cov").hide();
}

function closeCov() {
    $(".cov_register").hide();
}

//活动太火爆弹窗
function close_activity() {
    $(".cov_activity").show();
}

function activity() {
    $(".cov_activity").hide();
}

//反馈成功弹窗
function close_succeed() {
    $(".cov_succeed").show();
}

function succeed() {
    if ($(".con_succeed_test_Login").html() == "领专属礼包") {
        window.open("https://account.huaweicloud.com/usercenter/#/getCoupons?activityID=P1902141739108141NARGEA1I8FL99");
    }
    $(".cov_succeed").hide();
}

function dbt_succeed() {
    $(".cov_succeed").hide();
}

//弹窗end