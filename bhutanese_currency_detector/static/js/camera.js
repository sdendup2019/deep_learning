document.addEventListener('DOMContentLoaded', function() {
    const cameraToggle = document.getElementById('camera-toggle');
    const uploadToggle = document.getElementById('upload-toggle');
    const cameraSection = document.getElementById('camera-section');
    const uploadSection = document.getElementById('upload-section');
    const cameraFeed = document.getElementById('camera-feed');
    const captureBtn = document.getElementById('capture-btn');
    const retryBtn = document.getElementById('retry-btn');
    const canvas = document.getElementById('canvas');
    
    let stream = null;
    
    // Toggle between upload and camera sections
    cameraToggle.addEventListener('click', function() {
        cameraToggle.classList.add('active');
        uploadToggle.classList.remove('active');
        cameraSection.style.display = 'block';
        uploadSection.style.display = 'none';
        document.getElementById('results-section').style.display = 'none';
        
        // Start camera
        startCamera();
    });
    
    uploadToggle.addEventListener('click', function() {
        uploadToggle.classList.add('active');
        cameraToggle.classList.remove('active');
        uploadSection.style.display = 'block';
        cameraSection.style.display = 'none';
        
        // Stop camera if it's running
        stopCamera();
    });
    
    // Start camera function
    function startCamera() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(videoStream) {
                    stream = videoStream;
                    cameraFeed.srcObject = stream;
                    captureBtn.style.display = 'inline-block';
                    retryBtn.style.display = 'none';
                })
                .catch(function(error) {
                    console.error('Error accessing camera:', error);
                    alert('Could not access the camera. Please make sure you have given permission to use the camera.');
                });
        } else {
            alert('Sorry, your browser does not support camera access.');
        }
    }
    
    // Stop camera function
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => {
                track.stop();
            });
            stream = null;
        }
    }
    
    // Capture image function
    captureBtn.addEventListener('click', function() {
        // Set canvas dimensions to match video
        canvas.width = cameraFeed.videoWidth;
        canvas.height = cameraFeed.videoHeight;
        
        // Draw current video frame to canvas
        const context = canvas.getContext('2d');
        context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to base64 image
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Show loading and hide capture button
        document.getElementById('loading-section').style.display = 'block';
        captureBtn.style.display = 'none';
        retryBtn.style.display = 'inline-block';
        
        // Send image data to server
        sendImageToServer(imageData);
    });
    
    // Retry button
    retryBtn.addEventListener('click', function() {
        retryBtn.style.display = 'none';
        captureBtn.style.display = 'inline-block';
        document.getElementById('results-section').style.display = 'none';
    });
    
    // Send image to server for processing
    function sendImageToServer(imageData) {
        // Get CSRF token from cookie
        const csrftoken = getCookie('csrftoken');
        
        fetch('/detect_camera/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({
                image: imageData.split(',')[1]  // Remove data:image/jpeg;base64, part
            })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('loading-section').style.display = 'none';
            
            if (data.success) {
                displayResults(data);
            } else {
                alert('Error: ' + data.error);
                retryBtn.style.display = 'inline-block';
            }
        })
        .catch(error => {
            document.getElementById('loading-section').style.display = 'none';
            console.error('Error:', error);
            alert('An error occurred while processing the image.');
            retryBtn.style.display = 'inline-block';
        });
    }
    
    // Helper function to get CSRF token from cookies
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});