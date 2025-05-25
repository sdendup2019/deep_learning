document.addEventListener('DOMContentLoaded', function() {
    const currencyForm = document.getElementById('currency-form');
    
    if (currencyForm) {
        currencyForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading indicator
            document.getElementById('loading-section').style.display = 'block';
            document.getElementById('results-section').style.display = 'none';
            
            // Create FormData object
            const formData = new FormData(currencyForm);
            
            // Send form data to server
            fetch('/detect/', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading-section').style.display = 'none';
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                document.getElementById('loading-section').style.display = 'none';
                console.error('Error:', error);
                alert('An error occurred while processing the image.');
            });
        });
    }
});

// Function to display detection results
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    document.getElementById('currency-name').textContent = data.currency;
    document.getElementById('currency-description').textContent = data.description;
    
    // Set up audio if available
    const audioElement = document.getElementById('currency-audio');
    if (data.audio) {
        audioElement.src = `/static/audio/${data.audio}`;
        audioElement.load();
        resultsSection.style.display = 'block';
        
        // Play audio automatically
        audioElement.play().catch(e => {
            console.log('Auto-play prevented:', e);
        });
    } else {
        audioElement.src = '';
        resultsSection.style.display = 'block';
    }
}