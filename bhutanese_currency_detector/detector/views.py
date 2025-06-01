# #/detector/views.py

# from django.shortcuts import render, redirect
# from django.http import JsonResponse
# from .forms import CurrencyImageForm
# from .models import CurrencyImage
# from .ml_model.model_utils import detect_currency_from_image, process_base64_image
# from PIL import Image
# import json
# import base64
# import os

# def home(request):
#     form = CurrencyImageForm()
#     return render(request, 'home.html', {'form': form})

# def about(request):
#     # Define developers information
#     developers = [
#         {
#             'name': 'Dechen Peldon',
#             'role': 'ML Engineer',
#             'image': 'img/dechen.jpg',
#             'bio': 'Specialized in computer vision and deep learning.'
#         },
#         {
#             'name': 'Sonam Choden',
#             'role': 'Backend Developer',
#             'image': 'img/sc.jpeg',
#             'bio': 'Expert in Django and RESTful API development.'
#         },
#         {
#             'name': 'Sangay Dorji',
#             'role': 'Frontend Developer',
#             'image': 'img/b.jpeg',
#             'bio': 'Skilled in HTML, CSS, JavaScript and UI/UX design.'
#         },
#         # {
#         #     'name': 'Developer 4',
#         #     'role': 'Project Manager',
#         #     'image': 'img/dev4.jpg',
#         #     'bio': 'Experienced in agile methodology and project coordination.'
#         # }
#     ]
    
#     context = {
#         'developers': developers,
#         'about_text': 'This Bhutanese Currency Detection System is designed to help users identify Bhutanese currency notes through image recognition. The system utilizes a deep learning model trained on thousands of currency images to accurately identify and classify Bhutanese Ngultrum notes. This technology can be particularly useful for tourists, visually impaired individuals, and for automated currency counting systems.'
#     }
    
#     return render(request, 'about.html', context)

# def detect_currency(request):
#     if request.method == 'POST':
#         form = CurrencyImageForm(request.POST, request.FILES)
#         if form.is_valid():
#             # Save the uploaded image
#             currency_image = form.save(commit=False)
            
#             # Open the image using PIL
#             img = Image.open(currency_image.image)
            
#             # Detect currency
#             result = detect_currency_from_image(img)
            
#             # Save the result
#             currency_image.result = result['name']
#             currency_image.save()
            
#             # Return the result as JSON
#             return JsonResponse({
#                 'success': True,
#                 'currency': result['name'],
#                 'description': result['description'],
#                 'audio': result['audio']
#             })
#         else:
#             return JsonResponse({'success': False, 'error': 'Invalid form submission'})
    
#     return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})

# def detect_camera(request):
#     if request.method == 'POST':
#         try:
#             # Get the base64 image data from the request
#             data = json.loads(request.body)
#             base64_image = data.get('image')
            
#             if not base64_image:
#                 return JsonResponse({'success': False, 'error': 'No image data received'})
            
#             # Process the base64 image
#             pil_img = process_base64_image(base64_image)
            
#             # Detect currency
#             result = detect_currency_from_image(pil_img)
            
#             # Return the result as JSON
#             return JsonResponse({
#                 'success': True,
#                 'currency': result['name'],
#                 'description': result['description'],
#                 'audio': result['audio']
#             })
#         except Exception as e:
#             return JsonResponse({'success': False, 'error': str(e)})
    
#     return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})

# /detector/views.py

from django.shortcuts import render, redirect
from django.http import JsonResponse
from .forms import CurrencyImageForm
from .models import CurrencyImage
from .ml_model.model_utils import (
    detect_currency_from_image, 
    process_base64_image, 
    debug_model_predictions,
    get_model_info
)
from PIL import Image
import json
import base64
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    ]
    
    context = {
        'developers': developers,
        'about_text': 'This Bhutanese Currency Detection System is designed to help users identify Bhutanese currency notes through image recognition. The system utilizes a deep learning model trained on thousands of currency images to accurately identify and classify Bhutanese Ngultrum notes. This technology can be particularly useful for tourists, visually impaired individuals, and for automated currency counting systems.'
    }
    
    return render(request, 'about.html', context)

def detect_currency(request):
    if request.method == 'POST':
        try:
            logger.info("Starting currency detection from uploaded image")
            
            form = CurrencyImageForm(request.POST, request.FILES)
            if form.is_valid():
                # Save the uploaded image
                currency_image = form.save(commit=False)
                
                logger.info(f"Processing uploaded image: {currency_image.image.name}")
                
                # Open the image using PIL
                img = Image.open(currency_image.image)
                logger.info(f"Image opened successfully: {img.size}, mode: {img.mode}")
                
                # Detect currency with enhanced debugging
                result = detect_currency_from_image(img)
                logger.info(f"Detection result: {result}")
                
                # Save the result
                currency_image.result = result['name']
                currency_image.save()
                
                # Add debug information if confidence is low
                debug_info = {}
                if result.get('confidence', 0) < 0.5:
                    logger.warning(f"Low confidence detection: {result.get('confidence')}")
                    debug_info = debug_model_predictions(img)
                
                # Return the result as JSON
                response_data = {
                    'success': True,
                    'currency': result['name'],
                    'description': result['description'],
                    'audio': result['audio'],
                    'confidence': result.get('confidence', 0),
                    'predicted_class': result.get('predicted_class', -1),
                    'debug_info': debug_info
                }
                
                logger.info(f"Returning response: {response_data}")
                return JsonResponse(response_data)
            else:
                logger.error(f"Form validation failed: {form.errors}")
                return JsonResponse({'success': False, 'error': f'Invalid form submission: {form.errors}'})
                
        except Exception as e:
            logger.error(f"Error in detect_currency: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': f'Processing error: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})

def detect_camera(request):
    if request.method == 'POST':
        try:
            logger.info("Starting currency detection from camera image")
            
            # Get the base64 image data from the request
            data = json.loads(request.body)
            base64_image = data.get('image')
            
            if not base64_image:
                return JsonResponse({'success': False, 'error': 'No image data received'})
            
            logger.info(f"Received base64 image data, length: {len(base64_image)}")
            
            # Process the base64 image
            pil_img = process_base64_image(base64_image)
            logger.info(f"Processed base64 image: {pil_img.size}, mode: {pil_img.mode}")
            
            # Detect currency
            result = detect_currency_from_image(pil_img)
            logger.info(f"Detection result: {result}")
            
            # Add debug information if confidence is low
            debug_info = {}
            if result.get('confidence', 0) < 0.5:
                logger.warning(f"Low confidence detection: {result.get('confidence')}")
                debug_info = debug_model_predictions(pil_img)
            
            # Return the result as JSON
            response_data = {
                'success': True,
                'currency': result['name'],
                'description': result['description'],
                'audio': result['audio'],
                'confidence': result.get('confidence', 0),
                'predicted_class': result.get('predicted_class', -1),
                'debug_info': debug_info
            }
            
            logger.info(f"Returning response: {response_data}")
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Error in detect_camera: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': f'Processing error: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})

def debug_model(request):
    """
    Debug endpoint to check model status and test predictions
    """
    try:
        model_info = get_model_info()
        return JsonResponse({
            'success': True,
            'model_info': model_info
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

def test_model(request):
    """
    Test endpoint to verify model is working with a sample image
    """
    if request.method == 'POST':
        try:
            # Create a test image (solid color)
            from PIL import Image
            import numpy as np
            
            test_img = Image.new('RGB', (224, 224), color='red')
            result = detect_currency_from_image(test_img)
            
            return JsonResponse({
                'success': True,
                'test_result': result,
                'message': 'Model test completed'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})