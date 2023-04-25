from django.urls import path
from . import views

urlpatterns = [
    path("overlay/", views.home, name="home")
]