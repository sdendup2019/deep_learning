from django.contrib import admin
from .models import CurrencyImage

@admin.register(CurrencyImage)
class CurrencyImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'result', 'uploaded_at')
    list_filter = ('result', 'uploaded_at')
    search_fields = ('result',)