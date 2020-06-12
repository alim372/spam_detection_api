from rest_framework import routers
from predictions.views import userviewsets

router = routers.DefaultRouter()
router.register('user', userviewsets)