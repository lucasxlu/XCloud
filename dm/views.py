from django.shortcuts import render


# Create your views here.

def welcome(request):
    """
    welcome page for computer vision welcome
    :param request:
    :return:
    """
    return render(request, 'welcome.html')


def skindb(request):
    """
    query skin detailed information from Knowledge Base
    :param request:
    :return:
    """
    from dm.controllers import get_skin_info_from_kb_by_name

    return get_skin_info_from_kb_by_name(request)
