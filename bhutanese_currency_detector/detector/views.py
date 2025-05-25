from django.shortcuts import render, redirect
from django.http import JsonResponse
from .forms import CurrencyImageForm
from .models import CurrencyImage
from .ml_model.model_utils import detect_currency_from_image, process_base64_image
from PIL import Image
import json
import base64
import os

def home(request):
    form = CurrencyImageForm()
    return render(request, 'home.html', {'form': form})

def about(request):
    # Define developers information
    developers = [
        {
            'name': 'Dechen Peldon',
            'role': 'ML Engineer',
            'image': 'img/dechen.jpg',
            'bio': 'Specialized in computer vision and deep learning.'
        },
        {
            'name': 'Sonam Choden',
            'role': 'Backend Developer',
            'image': 'img/sc.jpeg',
            'bio': 'Expert in Django and RESTful API development.'
        },
        {
            'name': 'Sangay Dorji',
            'role': 'Frontend Developer',
            'image': 'img/b.jpeg',
            'bio': 'Skilled in HTML, CSS, JavaScript and UI/UX design.'
        },
        # {
        #     'name': 'Developer 4',
        #     'role': 'Project Manager',
        #     'image': 'img/dev4.jpg',
        #     'bio': 'Experienced in agile methodology and project coordination.'
        # }
    ]
    
    context = {
        'developers': developers,
        'about_text': 'This Bhutanese Currency Detection System is designed to help users identify Bhutanese currency notes through image recognition. The system utilizes a deep learning model trained on thousands of currency images to accurately identify and classify Bhutanese Ngultrum notes. This technology can be particularly useful for tourists, visually impaired individuals, and for automated currency counting systems.'
    }
    
    return render(request, 'about.html', context)

def detect_currency(request):
    if request.method == 'POST':
        form = CurrencyImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            currency_image = form.save(commit=False)
            
            # Open the image using PIL
            img = Image.open(currency_image.image)
            
            # Detect currency
            result = detect_currency_from_image(img)
            
            # Save the result
            currency_image.result = result['name']
            currency_image.save()
            
            # Return the result as JSON
            return JsonResponse({
                'success': True,
                'currency': result['name'],
                'description': result['description'],
                'audio': result['audio']
            })
        else:
            return JsonResponse({'success': False, 'error': 'Invalid form submission'})
    
    return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})

def detect_camera(request):
    if request.method == 'POST':
        try:
            # Get the base64 image data from the request
            data = json.loads(request.body)
            base64_image = data.get('image')
            
            if not base64_image:
                return JsonResponse({'success': False, 'error': 'No image data received'})
            
            # Process the base64 image
            pil_img = process_base64_image(base64_image)
            
            # Detect currency
            result = detect_currency_from_image(pil_img)
            
            # Return the result as JSON
            return JsonResponse({
                'success': True,
                'currency': result['name'],
                'description': result['description'],
                'audio': result['audio']
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})