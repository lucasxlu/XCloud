$(function () {
    // 1.初始化Table
    var oTable = new TableInit();
    oTable.Init();
});

var TableInit = function () {
    var oTableInit = new Object();
    // 初始化Table
    oTableInit.Init = function () {
        $('#tableinfo').bootstrapTable(
            {
                url: '/api/hzau/job/view', // 请求后台的URL（*）
                method: 'get', // 请求方式（*）
                toolbar: '#toolbar', // 工具按钮用哪个容器
                striped: false, // 是否显示行间隔色
                cache: false, // 是否使用缓存，默认为true，所以一般情况下需要设置一下这个属性（*）
                search: true,
                pagination: true, // 是否显示分页（*）
                sortable: true, // 是否启用排序
                sortOrder: "asc", // 排序方式
                queryParams: oTableInit.queryParams,// 传递参数（*）
                sidePagination: "server", // 分页方式：client客户端分页，server服务端分页（*）
                pageNumber: 1, // 初始化加载第一页，默认第一页
                pageSize: 10, // 每页的记录行数（*）
                pageList: [10, 25, 50, 100], // 可供选择的每页的行数（*）
                strictSearch : true,
                clickToSelect: true, // 是否启用点击选中行
                // height: 460, //行高，如果没有设置height属性，表格自动根据记录条数觉得表格高度
                uniqueId: "id", // 每一行的唯一标识，一般为主键列
                cardView: false, // 是否显示详细视图
                detailView: false, // 是否显示父子表
                showRefresh: true,
                showColumns: true,
                showToggle: true,
                rowStyle: function (row, index) {
                    var classes = ['success', 'info', 'warning', 'danger', 'active',];

                    if ((index + 1) % 3 === 0 && (index + 1) / 3 < classes.length) {
                        return {
                            classes: classes[(index + 1) / 3]
                        };
                    }
                    return {};
                },
                formatLoadingMessage: function () {
                    return "loading, please wait...";
                },
                columns: [
                    /* {
                         checkbox : true
                     },*/
                    {
                        field: 'jobname',
                        title: 'jobname',
                        align: 'center',
                        formatter: function (value, row, index) {
                            return '<a href="' + row.detailurl + '" target="_blank">' + value + '</a>';
                        }
                    },
                    {
                        field: 'company',
                        title: 'company',
                        align: 'center'
                    },
                    /* {
                         field : 'logopath',
                         title : 'logo',
                         align : 'center',

                     },*/
                    {
                        field: 'jobtype',
                        title: 'jobtype',
                        align: 'center'
                    },
                    {
                        field: 'publishdate',
                        title: 'publishdate',
                        align: 'center',
                        sortable : true,
                        formatter: function (value, row, index) {
                            // return getLocalTime(value, 'yyyy-MM-dd HH:mm:ss');
                            return getLocalTime(value);
                        }
                    },
                    {
                        field: 'location',
                        title: 'location',
                        align: 'center'
                    }, {
                        field: 'applynum',
                        title: 'applynum',
                        align: 'center',
                        sortable : true
                    }
                    ,
                    {
                        title: 'handle',
                        align: 'center',
                        width: '70px',
                        formatter: function (value, row, index) {
                            return '<a class="btn blue btn-xs" style="width:54px;" onclick="deleteCurRow(' + row.id + ')">delete</button>';
                        }
                    }]
            });
    };
    // 得到查询的参数
    oTableInit.queryParams = function (params) {
        console.log(params)
        var temp = { // 这里的键的名字和控制器的变量名必须一直，这边改动，控制器也需要改成一样的
            limit: params.limit, // 页面大小
            offset: params.offset, // 页码
            sort:params.sort,
            order:params.order,
            search:params.search,
            jobType: $("#jobType option:selected").val()
        };
        return temp;
    };
    return oTableInit;
};


//查询方法
function doSearch() {
    $('#tableinfo').bootstrapTable('refresh');
}

//重置
function doReset() {
    $('#jobType').val('');
    $('#tableinfo').bootstrapTable('refresh');
}

function deleteCurRow(id) {
    layer.confirm("<i class='fa fa-question-circle'></i>" + "confirm to delete?", {btn: ['ok', 'cancel']},
        function () {
            $.ajax({
                url: "/api/hzau/job/delete",    //请求的url地址
                dataType: "json",   //返回格式为json
                data: {"id": id},    //参数值
                type: "POST",   //请求方式
                success: function (req) {//请求成功时处理
                    layer.msg("删除成功", {icon: 1});
                    $('#tableinfo').bootstrapTable('refresh');
                }
            });
        })
}

