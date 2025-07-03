import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import urllib.parse
import warnings
import datetime
import json
warnings.filterwarnings('ignore')

# Sample dataset - In real scenario, you'd load from a CSV file
# Creating synthetic data for demonstration
def create_sample_data():
    """Create sample phishing and legitimate URLs for training"""
    
    # Sample legitimate URLs
    legitimate_urls = [
        "https://www.google.com",
        "https://www.facebook.com",
        "https://www.amazon.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.linkedin.com",
        "https://www.youtube.com",
        "https://www.twitter.com",
        "https://www.instagram.com",
        "https://www.reddit.com",
        "https://www.wikipedia.org",
        "https://www.netflix.com",
        "https://www.paypal.com",
        "https://www.ebay.com",
        "https://www.adobe.com",
        "https://www.dropbox.com",
        "https://www.spotify.com",
        "https://www.uber.com"
    ]
    
    # Sample phishing URLs (simulated - not real phishing sites)
    phishing_urls = [
        "http://paypal-security-update.com/login",
        "https://amazon-prize-winner.net/claim",
        "http://google-verification.tk/verify",
        "https://facebook-security-alert.ml/login",
        "http://apple-id-locked.cf/unlock",
        "https://microsoft-support-team.ga/help",
        "http://bank-security-alert.tk/verify",
        "https://urgent-paypal-verification.ml/login",
        "http://amazon-customer-service.cf/refund",
        "https://secure-banking-update.ga/login",
        "http://netflix-account-suspended.tk/reactivate",
        "https://google-account-recovery.ml/restore",
        "http://ebay-seller-verification.cf/verify",
        "https://instagram-security-check.ga/login",
        "http://linkedin-profile-verification.tk/confirm",
        "https://twitter-account-suspended.ml/appeal",
        "http://dropbox-storage-full.cf/upgrade",
        "https://spotify-premium-free.ga/claim",
        "http://uber-ride-refund.tk/process",
        "https://adobe-license-expired.ml/renew"
    ]
    
    # Create DataFrame
    urls = legitimate_urls + phishing_urls
    labels = [0] * len(legitimate_urls) + [1] * len(phishing_urls)  # 0 = legitimate, 1 = phishing
    
    df = pd.DataFrame({
        'url': urls,
        'label': labels
    })
    
    return df

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.suspicious_keywords = [
            'verify', 'account', 'update', 'confirm', 'login', 'secure', 'banking',
            'alert', 'suspended', 'locked', 'verification', 'security', 'urgent',
            'winner', 'prize', 'claim', 'free', 'bonus', 'gift', 'reward'
        ]
        
        self.top_domains = [
            'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 
            'apple.com', 'paypal.com', 'ebay.com', 'netflix.com'
        ]
    
    def extract_features(self, url):
        """Extract features from a single URL"""
        features = {}
        
        # Basic URL characteristics
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equal_signs'] = url.count('=')
        features['num_ampersands'] = url.count('&')
        features['num_percent_signs'] = url.count('%')
        features['num_digits'] = sum(c.isdigit() for c in url)
        
        # Protocol analysis
        features['is_https'] = 1 if url.startswith('https://') else 0
        features['is_http'] = 1 if url.startswith('http://') else 0
        
        # Domain analysis
        try:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc.lower()
            
            features['domain_length'] = len(domain)
            features['num_subdomains'] = len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0
            features['has_ip_address'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0
            
            # Check for suspicious TLDs
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.online', '.click', '.download']
            features['suspicious_tld'] = 1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0
            
            # Check if domain contains legitimate brand names
            features['contains_brand_name'] = 0
            for brand in self.top_domains:
                brand_name = brand.split('.')[0]
                if brand_name in domain and not domain.endswith(brand):
                    features['contains_brand_name'] = 1
                    break
            
        except:
            features['domain_length'] = 0
            features['num_subdomains'] = 0
            features['has_ip_address'] = 0
            features['suspicious_tld'] = 0
            features['contains_brand_name'] = 0
        
        # Path analysis
        path = urllib.parse.urlparse(url).path
        features['path_length'] = len(path)
        features['num_path_segments'] = len([seg for seg in path.split('/') if seg])
        
        # Query parameters
        query = urllib.parse.urlparse(url).query
        features['query_length'] = len(query)
        features['num_query_params'] = len(urllib.parse.parse_qs(query))
        
        # Suspicious keywords
        url_lower = url.lower()
        features['num_suspicious_keywords'] = sum(1 for keyword in self.suspicious_keywords if keyword in url_lower)
        
        # Character distribution
        features['ratio_digits'] = features['num_digits'] / len(url) if len(url) > 0 else 0
        features['ratio_special_chars'] = (features['num_hyphens'] + features['num_underscores']) / len(url) if len(url) > 0 else 0
        
        # URL complexity
        features['entropy'] = self.calculate_entropy(url)
        
        return features
    
    def calculate_entropy(self, string):
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0
        
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = -sum([p * np.log2(p) for p in prob])
        return entropy
    
    def extract_features_batch(self, urls):
        """Extract features for multiple URLs"""
        features_list = []
        for url in urls:
            features = self.extract_features(url)
            features_list.append(features)
        
        return pd.DataFrame(features_list)

class PhishingDetector:
    """Main phishing detection class"""
    
    def __init__(self):
        self.feature_extractor = URLFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Extracting features from URLs...")
        
        # Extract features
        features_df = self.feature_extractor.extract_features_batch(df['url'])
        
        # Combine with labels
        X = features_df
        y = df['label']
        
        print(f"Extracted {len(X.columns)} features from {len(X)} URLs")
        print(f"Features: {list(X.columns)}")
        
        return X, y
    
    def train_model(self, X, y):
        """Train the phishing detection model"""
        print("\nTraining model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Training Complete!")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return accuracy, y_test, y_pred
    
    def predict_url(self, url):
        """Predict if a single URL is phishing or legitimate"""
        if self.model is None:
            raise Exception("Model not trained yet. Please train the model first.")
        
        # Extract features
        features = self.feature_extractor.extract_features(url)
        features_df = pd.DataFrame([features])
        
        # Ensure all features are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training data
        features_df = features_df[self.feature_columns]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'url': url,
            'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
            'confidence': max(probability),
            'phishing_probability': probability[1],
            'legitimate_probability': probability[0]
        }
        
        return result
    
    def batch_predict(self, urls):
        """Predict multiple URLs"""
        results = []
        for url in urls:
            result = self.predict_url(url)
            results.append(result)
        return results

class HTMLReportGenerator:
    """Generate beautiful HTML reports for phishing detection results"""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_html_report(self, detector, accuracy, y_test, y_pred, test_results, feature_importance):
        """Generate comprehensive HTML report"""
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Prepare data for report
        self.report_data = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'test_results': test_results,
            'feature_importance': feature_importance.to_dict('records'),
            'total_urls_tested': len(test_results),
            'phishing_detected': len([r for r in test_results if r['prediction'] == 'Phishing']),
            'legitimate_detected': len([r for r in test_results if r['prediction'] == 'Legitimate'])
        }
        
        html_content = self.create_html_template()
        
        # Save report
        filename = f"phishing_detection_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n📊 HTML Report generated: {filename}")
        return filename
    
    def create_html_template(self):
        """Create HTML template with embedded CSS and JavaScript"""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing URL Detection Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 5px solid #3498db;
        }}
        
        .section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            display: flex;
            align-items: center;
        }}
        
        .section h2::before {{
            content: "📊";
            margin-right: 10px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 10px;
        }}
        
        .metric-label {{
            font-size: 1.1em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .url-result {{
            background: white;
            margin-bottom: 15px;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #ddd;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .url-result.phishing {{
            border-left-color: #e74c3c;
            background: linear-gradient(135deg, #fff5f5 0%, #ffeaea 100%);
        }}
        
        .url-result.legitimate {{
            border-left-color: #27ae60;
            background: linear-gradient(135deg, #f0fff4 0%, #e8f5e8 100%);
        }}
        
        .url-text {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            word-break: break-all;
        }}
        
        .prediction {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 8px;
        }}
        
        .prediction.phishing {{
            color: #e74c3c;
        }}
        
        .prediction.legitimate {{
            color: #27ae60;
        }}
        
        .confidence-bar {{
            background: #ecf0f1;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        
        .confidence-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        
        .confidence-fill.phishing {{
            background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%);
        }}
        
        .confidence-fill.legitimate {{
            background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
        }}
        
        .confidence-text {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .feature-importance {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .feature-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .feature-item:last-child {{
            border-bottom: none;
        }}
        
        .feature-name {{
            font-weight: 500;
            color: #2c3e50;
        }}
        
        .feature-bar {{
            width: 200px;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin-left: 20px;
        }}
        
        .feature-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
            border-radius: 4px;
        }}
        
        .confusion-matrix {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            max-width: 400px;
            margin: 20px auto;
        }}
        
        .cm-cell {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .cm-cell.correct {{
            background: linear-gradient(135deg, #d5f4e6 0%, #c8e6c9 100%);
        }}
        
        .cm-cell.incorrect {{
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        }}
        
        .cm-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .cm-label {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .timestamp {{
            opacity: 0.8;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .confusion-matrix {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Phishing URL Detection Report</h1>
            <p>Machine Learning Model Performance Analysis</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>Model Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{self.report_data['accuracy']:.3f}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.report_data['total_urls_tested']}</div>
                        <div class="metric-label">URLs Tested</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.report_data['phishing_detected']}</div>
                        <div class="metric-label">Phishing Detected</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.report_data['legitimate_detected']}</div>
                        <div class="metric-label">Legitimate URLs</div>
                    </div>
                </div>
                
                <h3>Confusion Matrix</h3>
                <div class="confusion-matrix">
                    <div class="cm-cell correct">
                        <div class="cm-value">{self.report_data['confusion_matrix'][0][0]}</div>
                        <div class="cm-label">True Legitimate</div>
                    </div>
                    <div class="cm-cell incorrect">
                        <div class="cm-value">{self.report_data['confusion_matrix'][0][1]}</div>
                        <div class="cm-label">False Phishing</div>
                    </div>
                    <div class="cm-cell incorrect">
                        <div class="cm-value">{self.report_data['confusion_matrix'][1][0]}</div>
                        <div class="cm-label">False Legitimate</div>
                    </div>
                    <div class="cm-cell correct">
                        <div class="cm-value">{self.report_data['confusion_matrix'][1][1]}</div>
                        <div class="cm-label">True Phishing</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>URL Analysis Results</h2>
                {self.generate_url_results_html()}
            </div>
            
            <div class="section">
                <h2>Feature Importance Analysis</h2>
                <div class="feature-importance">
                    {self.generate_feature_importance_html()}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Report generated on {self.report_data['timestamp']}</p>
            <p class="timestamp">Phishing URL Detector - Machine Learning Model</p>
        </div>
    </div>
    
    <script>
        // Add smooth animations
        document.addEventListener('DOMContentLoaded', function() {{
            const cards = document.querySelectorAll('.metric-card, .url-result');
            cards.forEach((card, index) => {{
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                setTimeout(() => {{
                    card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }}, index * 100);
            }});
        }});
    </script>
</body>
</html>
        """
        
        return html_template
    
    def generate_url_results_html(self):
        """Generate HTML for URL analysis results"""
        html = ""
        for result in self.report_data['test_results']:
            css_class = result['prediction'].lower()
            confidence_percent = result['confidence'] * 100
            
            html += f"""
            <div class="url-result {css_class}">
                <div class="url-text">{result['url']}</div>
                <div class="prediction {css_class}">
                    {'🚨 ' if result['prediction'] == 'Phishing' else '✅ '}{result['prediction']}
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill {css_class}" style="width: {confidence_percent}%"></div>
                </div>
                <div class="confidence-text">
                    Confidence: {confidence_percent:.1f}% | 
                    Phishing Probability: {result['phishing_probability']:.3f}
                </div>
            </div>
            """
        
        return html
    
    def generate_feature_importance_html(self):
        """Generate HTML for feature importance"""
        html = ""
        max_importance = max([f['importance'] for f in self.report_data['feature_importance'][:10]])
        
        for feature in self.report_data['feature_importance'][:10]:
            bar_width = (feature['importance'] / max_importance) * 100
            html += f"""
            <div class="feature-item">
                <span class="feature-name">{feature['feature'].replace('_', ' ').title()}</span>
                <div class="feature-bar">
                    <div class="feature-bar-fill" style="width: {bar_width}%"></div>
                </div>
                <span class="feature-value">{feature['importance']:.4f}</span>
            </div>
            """
        
        return html

def main():
    """Main function to run the phishing detector"""
    print("=" * 60)
    print("PHISHING URL DETECTOR")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample dataset...")
    df = create_sample_data()
    print(f"Dataset created with {len(df)} URLs")
    print(f"Legitimate URLs: {len(df[df['label'] == 0])}")
    print(f"Phishing URLs: {len(df[df['label'] == 1])}")
    
    # Initialize detector
    detector = PhishingDetector()
    
    # Prepare data
    X, y = detector.prepare_data(df)
    
    # Train model
    accuracy, y_test, y_pred = detector.train_model(X, y)
    
    # Test with new URLs
    print("\n" + "=" * 60)
    print("TESTING WITH NEW URLs")
    print("=" * 60)
    
    test_urls = [
        "https://www.google.com/search?q=python",
        "http://paypal-verification-urgent.tk/login",
        "https://www.amazon.com/products",
        "https://facebook-security-alert.ml/verify",
        "https://www.microsoft.com/office",
        "http://secure-banking-update.ga/login",
        "https://www.github.com/repositories",
        "https://urgent-paypal-verification.ml/login"
    ]
    
    test_results = []
    for url in test_urls:
        result = detector.predict_url(url)
        test_results.append(result)
        print(f"\nURL: {result['url']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Phishing Probability: {result['phishing_probability']:.4f}")
    
    # Generate feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': detector.feature_columns,
        'importance': detector.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Generate HTML Report
    print("\n" + "=" * 60)
    print("GENERATING HTML REPORT")
    print("=" * 60)
    
    report_generator = HTMLReportGenerator()
    report_filename = report_generator.generate_html_report(
        detector, accuracy, y_test, y_pred, test_results, feature_importance
    )
    
    print(f"✅ Report saved as: {report_filename}")
    print("📂 Open the HTML file in your browser to view the detailed report!")
    
    print("\n" + "=" * 60)
    print("PHISHING DETECTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
