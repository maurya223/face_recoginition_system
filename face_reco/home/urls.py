from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register, name="register"),
    path("login/", views.login, name="login"),
    path("attendance/", views.attendance, name="attendance"),
    path("get_attendance/", views.get_attendance, name="get_attendance"),
    path(
        "mark_attendance/", views.mark_attendance, name="mark_attendance"
    ),  # image-based
    path(
        "mark_attendance_manual/",
        views.mark_attendance_manual,
        name="mark_attendance_manual",
    ),  # name-based
    path("recognize_face/", views.ajax_recognize_face, name="recognize_face"),
    path("register-face/", views.register_face, name="register_face"),
    path("login-face/", views.login_face, name="login_face"),
    path("recognize/", views.ajax_recognize_face),
]
