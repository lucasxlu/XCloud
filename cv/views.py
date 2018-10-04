from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.


def welcome(request):
    """
    test page for computer vision welcome
    :param request:
    :return:
    """
    return HttpResponse("Hello, welcome to computer vision section.")
