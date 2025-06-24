"""
Enhanced Result Analyzer
Advanced analysis module using only core ML libraries for maximum compatibility
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import re
import json
from datetime import datetime
import math

# Core ML imports (always available)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

logger = logging.getLogger(__name__)

class EnhancedResultAnalyzer:
    """Enhanced result analyzer using proven ML algorithms."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.15, random_state=42)
        
        # Security and intelligence patterns
        self.security_indicators = self._load_security_indicators()
        self.file_risk_scores = self._load_file_risk_scores()
        self.technology_patterns = self._load_technology_patterns()
        
    def _load_security_indicators(self) -> Dict[str, float]:
        """Load security indicators with risk scores."""
        return {
            # High risk indicators
            'password': 0.95,
            'passwd': 0.95,
            'secret': 0.9,
            'api_key': 0.9,
            'token': 0.85,
            'credential': 0.85,
            'admin': 0.8,
            'administrator': 0.8,
            'root': 0.8,
            'backup': 0.75,
            'config': 0.7,
            'configuration': 0.7,
            'database': 0.7,
            'dump': 0.7,
            
            # Medium risk indicators
            'login': 0.6,
            'auth': 0.6,
            'private': 0.6,
            'internal': 0.55,
            'debug': 0.5,
            'test': 0.45,
            'staging': 0.4,
            'dev': 0.35,
            
            # Vulnerability indicators
            'vulnerability': 0.85,
            'exploit': 0.8,
            'cve': 0.75,
            'security': 0.5,
            'patch': 0.4,
            
            # Data exposure indicators
            'exposed': 0.8,
            'leaked': 0.8,
            'public': 0.6,
            'directory listing': 0.7,
            'index of': 0.7
        }
    
    def _load_file_risk_scores(self) -> Dict[str, float]:
        """Load file extension risk scores."""
        return {
            # Database files
            '.sql': 0.95,
            '.db': 0.9,
            '.sqlite': 0.85,
            '.mdb': 0.8,
            
            # Configuration files
            '.config': 0.9,
            '.conf': 0.85,
            '.ini': 0.8,
            '.cfg': 0.75,
            '.xml': 0.6,
            '.json': 0.6,
            '.yaml': 0.6,
            '.yml': 0.6,
            
            # Backup and archive files
            '.backup': 0.9,
            '.bak': 0.85,
            '.old': 0.7,
            '.zip': 0.6,
            '.tar': 0.6,
            '.gz': 0.5,
            
            # Log files
            '.log': 0.75,
            '.logs': 0.75,
            '.trace': 0.7,
            
            # Script files
            '.sh': 0.65,
            '.bat': 0.65,
            '.cmd': 0.65,
            '.ps1': 0.7,
            
            # Source code (potential for secrets)
            '.php': 0.5,
            '.asp': 0.5,
            '.jsp': 0.5,
            '.py': 0.4,
            '.rb': 0.4,
            '.js': 0.4,
            
            # Document files
            '.pdf': 0.4,
            '.doc': 0.35,
            '.docx': 0.35,
            '.xls': 0.4,
            '.xlsx': 0.4,
            '.txt': 0.3
        }
    
    def _load_technology_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load technology detection patterns."""
        return {
            'wordpress': {
                'indicators': ['wp-content', 'wp-admin', 'wp-includes', 'wordpress'],
                'risk_multiplier': 1.2,
                'common_issues': ['outdated plugins', 'weak passwords', 'file permissions']
            },
            'drupal': {
                'indicators': ['drupal', 'sites/default', '/node/', 'modules/'],
                'risk_multiplier': 1.1,
                'common_issues': ['sql injection', 'privilege escalation']
            },
            'joomla': {
                'indicators': ['joomla', 'administrator/', 'components/', 'templates/'],
                'risk_multiplier': 1.15,
                'common_issues': ['rce vulnerabilities', 'weak authentication']
            },
            'apache': {
                'indicators': ['apache', 'httpd', '.htaccess', 'server-status'],
                'risk_multiplier': 1.0,
                'common_issues': ['misconfigurations', 'directory traversal']
            },
            'nginx': {
                'indicators': ['nginx', 'nginx.conf'],
                'risk_multiplier': 0.9,
                'common_issues': ['misconfigurations', 'proxy issues']
            },
            'iis': {
                'indicators': ['iis', 'aspnet', 'web.config'],
                'risk_multiplier': 1.1,
                'common_issues': ['aspnet vulnerabilities', 'misconfigurations']
            }
        }
    
    def analyze_results_enhanced(self, results: List[Dict[str, Any]], target: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform enhanced analysis on search results."""
        if not results:
            return results
        
        logger.info(f"Analyzing {len(results)} results with enhanced ML algorithms")
        
        try:
            # Extract text features
            text_features = self._extract_text_features(results)
            
            # Calculate multiple scoring dimensions
            relevance_scores = self._calculate_relevance_scores(results, target)
            security_scores = self._calculate_security_scores(results)
            anomaly_scores = self._detect_result_anomalies(results, text_features)
            technology_scores = self._analyze_technology_indicators(results)
            
            # Enhanced sentiment and context analysis
            sentiment_scores = self._analyze_sentiment_context(results)
            
            # Combine all analyses
            enhanced_results = []
            for i, result in enumerate(results):
                enhanced_result = result.copy()
                
                # Add ML-derived scores
                enhanced_result.update({
                    'ml_relevance_score': relevance_scores[i],
                    'security_risk_score': security_scores[i],
                    'anomaly_score': anomaly_scores[i],
                    'technology_score': technology_scores[i],
                    'sentiment_score': sentiment_scores[i],
                    'composite_ml_score': self._calculate_composite_score(
                        relevance_scores[i], security_scores[i], 
                        anomaly_scores[i], technology_scores[i]
                    )
                })
                
                # Enhanced intelligence indicators
                enhanced_result['ml_intelligence_indicators'] = self._extract_ml_indicators(result)
                
                # Risk categorization
                enhanced_result['ml_risk_category'] = self._categorize_risk(enhanced_result)
                
                # Confidence estimation
                enhanced_result['ml_confidence'] = self._estimate_confidence(enhanced_result)
                
                enhanced_results.append(enhanced_result)
            
            # Perform clustering analysis
            clustered_results = self._cluster_similar_results(enhanced_results)
            
            # Final ranking
            ranked_results = self._rank_results_ml(clustered_results, context)
            
            logger.info(f"Enhanced analysis completed. Average ML confidence: {np.mean([r.get('ml_confidence', 0.5) for r in ranked_results]):.3f}")
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            return results
    
    def _extract_text_features(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """Extract TF-IDF features from result text."""
        texts = []
        for result in results:
            text = f"{result.get('title', '')} {result.get('snippet', '')} {result.get('url', '')}"
            texts.append(text.lower())
        
        try:
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()
        except Exception as e:
            logger.debug(f"TF-IDF feature extraction failed: {e}")
            return np.zeros((len(results), 100))
    
    def _calculate_relevance_scores(self, results: List[Dict[str, Any]], target: str) -> List[float]:
        """Calculate enhanced relevance scores."""
        scores = []
        target_lower = target.lower()
        target_parts = re.split(r'[.\-_]', target_lower)
        
        for result in results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            url = result.get('url', '').lower()
            
            # Direct target matching
            title_match = self._calculate_text_similarity(title, target_lower)
            snippet_match = self._calculate_text_similarity(snippet, target_lower)
            url_match = self._calculate_text_similarity(url, target_lower)
            
            # Part-based matching
            part_matches = []
            for part in target_parts:
                if len(part) > 2:  # Skip very short parts
                    part_score = (
                        (1.0 if part in title else 0.0) * 0.4 +
                        (1.0 if part in snippet else 0.0) * 0.3 +
                        (1.0 if part in url else 0.0) * 0.3
                    )
                    part_matches.append(part_score)
            
            avg_part_match = np.mean(part_matches) if part_matches else 0.0
            
            # Combine scores with weights
            relevance_score = (
                title_match * 0.35 +
                snippet_match * 0.25 +
                url_match * 0.2 +
                avg_part_match * 0.2
            )
            
            scores.append(min(relevance_score, 1.0))
        
        return scores
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using multiple methods."""
        if not text1 or not text2:
            return 0.0
        
        # Exact match bonus
        if text2 in text1:
            return 1.0
        
        # Jaccard similarity for word-level comparison
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Character-level similarity
        char_similarity = self._calculate_char_similarity(text1, text2)
        
        # Weighted combination
        return jaccard_sim * 0.7 + char_similarity * 0.3
    
    def _calculate_char_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity."""
        if not text1 or not text2:
            return 0.0
        
        # Simple character overlap
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())
        
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_security_scores(self, results: List[Dict[str, Any]]) -> List[float]:
        """Calculate security risk scores for results."""
        scores = []
        
        for result in results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            url = result.get('url', '').lower()
            content = f"{title} {snippet} {url}"
            
            security_score = 0.0
            
            # Check security indicators
            for indicator, weight in self.security_indicators.items():
                if indicator in content:
                    security_score += weight * 0.2
            
            # Check file extensions
            for ext, risk in self.file_risk_scores.items():
                if ext in url:
                    security_score += risk * 0.3
            
            # Pattern-based detection
            patterns = {
                r'password\s*[:=]\s*\w+': 0.9,
                r'api[_-]?key\s*[:=]\s*[\w\-]+': 0.85,
                r'secret\s*[:=]\s*\w+': 0.8,
                r'token\s*[:=]\s*[\w\-]+': 0.75,
                r'config\s*[:=]': 0.6,
                r'admin\s*[:=]': 0.7,
                r'root\s*[:=]': 0.8,
                r'backup\s*[:=]': 0.65
            }
            
            for pattern, risk in patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    security_score += risk * 0.15
            
            # URL structure analysis
            suspicious_paths = [
                'admin', 'administrator', 'private', 'internal', 'config',
                'backup', 'temp', 'tmp', 'test', 'staging', 'dev'
            ]
            
            for path in suspicious_paths:
                if f'/{path}/' in url or f'/{path}?' in url:
                    security_score += 0.3
            
            scores.append(min(security_score, 1.0))
        
        return scores
    
    def _detect_result_anomalies(self, results: List[Dict[str, Any]], text_features: np.ndarray) -> List[float]:
        """Detect anomalous results that might be interesting."""
        if len(results) < 3:
            return [0.5] * len(results)
        
        try:
            # Prepare features for anomaly detection
            features = []
            for i, result in enumerate(results):
                title_len = len(result.get('title', ''))
                snippet_len = len(result.get('snippet', ''))
                url_len = len(result.get('url', ''))
                
                # Count numeric characters (might indicate versions, IDs, etc.)
                title_nums = len(re.findall(r'\d', result.get('title', '')))
                url_nums = len(re.findall(r'\d', result.get('url', '')))
                
                # Special characters
                special_chars = len(re.findall(r'[^\w\s]', result.get('url', '')))
                
                feature_vector = [
                    title_len, snippet_len, url_len,
                    title_nums, url_nums, special_chars
                ]
                
                # Add text feature summary
                if i < len(text_features):
                    text_feature_sum = np.sum(text_features[i])
                    feature_vector.append(text_feature_sum)
                else:
                    feature_vector.append(0.0)
                
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Detect anomalies
            anomaly_predictions = self.anomaly_detector.fit_predict(features_scaled)
            
            # Convert to scores (anomalies get higher scores)
            anomaly_scores = []
            for pred in anomaly_predictions:
                if pred == -1:  # Anomaly
                    anomaly_scores.append(0.8)
                else:  # Normal
                    anomaly_scores.append(0.2)
            
            return anomaly_scores
            
        except Exception as e:
            logger.debug(f"Anomaly detection failed: {e}")
            return [0.5] * len(results)
    
    def _analyze_technology_indicators(self, results: List[Dict[str, Any]]) -> List[float]:
        """Analyze technology indicators in results."""
        scores = []
        
        for result in results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            url = result.get('url', '').lower()
            content = f"{title} {snippet} {url}"
            
            tech_score = 0.0
            detected_techs = []
            
            # Check for technology patterns
            for tech, tech_info in self.technology_patterns.items():
                indicators = tech_info['indicators']
                risk_multiplier = tech_info['risk_multiplier']
                
                tech_matches = 0
                for indicator in indicators:
                    if indicator in content:
                        tech_matches += 1
                
                if tech_matches > 0:
                    tech_confidence = min(tech_matches / len(indicators), 1.0)
                    tech_contribution = tech_confidence * risk_multiplier * 0.3
                    tech_score += tech_contribution
                    detected_techs.append(tech)
            
            # Version detection (might indicate outdated software)
            version_patterns = [
                r'v?\d+\.\d+\.\d+',
                r'version\s+\d+\.\d+',
                r'ver\s+\d+\.\d+'
            ]
            
            for pattern in version_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    tech_score += 0.2
                    break
            
            # Store detected technologies in result
            result['detected_technologies'] = detected_techs
            
            scores.append(min(tech_score, 1.0))
        
        return scores
    
    def _analyze_sentiment_context(self, results: List[Dict[str, Any]]) -> List[float]:
        """Analyze sentiment and context of results."""
        scores = []
        
        for result in results:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            # Combine title and snippet for analysis
            text = f"{title} {snippet}"
            
            if not text.strip():
                scores.append(0.5)
                continue
            
            try:
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                
                # Sentiment polarity (-1 to 1)
                polarity = blob.sentiment.polarity
                
                # Convert to risk-weighted score
                # Negative sentiment might indicate problems/issues
                if polarity < -0.1:  # Negative sentiment
                    sentiment_score = 0.7 + abs(polarity) * 0.3  # Higher score for negative sentiment
                elif polarity > 0.1:  # Positive sentiment
                    sentiment_score = 0.3 + polarity * 0.2  # Lower score for positive sentiment
                else:  # Neutral sentiment
                    sentiment_score = 0.5
                
                scores.append(min(sentiment_score, 1.0))
                
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")
                scores.append(0.5)
        
        return scores
    
    def _calculate_composite_score(self, relevance: float, security: float, anomaly: float, technology: float) -> float:
        """Calculate composite ML score."""
        # Weighted combination of different score dimensions
        composite = (
            relevance * 0.3 +      # Relevance is important
            security * 0.35 +      # Security findings are crucial
            anomaly * 0.2 +        # Anomalies might reveal interesting findings
            technology * 0.15      # Technology detection adds context
        )
        
        return min(composite, 1.0)
    
    def _extract_ml_indicators(self, result: Dict[str, Any]) -> List[str]:
        """Extract ML-derived intelligence indicators."""
        indicators = []
        
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        url = result.get('url', '').lower()
        content = f"{title} {snippet} {url}"
        
        # Security-related indicators
        for indicator in self.security_indicators.keys():
            if indicator in content:
                indicators.append(f"security_{indicator}")
        
        # File type indicators
        for ext in self.file_risk_scores.keys():
            if ext in url:
                indicators.append(f"filetype_{ext.replace('.', '')}")
        
        # Technology indicators
        detected_techs = result.get('detected_technologies', [])
        for tech in detected_techs:
            indicators.append(f"technology_{tech}")
        
        # Pattern-based indicators
        patterns = {
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b': 'ip_address',
            r'\b[A-Fa-f0-9]{32}\b': 'md5_hash',
            r'\b[A-Fa-f0-9]{40}\b': 'sha1_hash',
            r'\b[A-Fa-f0-9]{64}\b': 'sha256_hash',
            r'[a-zA-Z0-9]{20,}': 'long_string',
            r'[A-Z0-9]{8,}': 'uppercase_string'
        }
        
        for pattern, indicator_name in patterns.items():
            if re.search(pattern, content):
                indicators.append(indicator_name)
        
        return list(set(indicators))  # Remove duplicates
    
    def _categorize_risk(self, result: Dict[str, Any]) -> str:
        """Categorize risk level based on ML scores."""
        security_score = result.get('security_risk_score', 0.0)
        anomaly_score = result.get('anomaly_score', 0.0)
        composite_score = result.get('composite_ml_score', 0.0)
        
        # High risk criteria
        if (security_score > 0.7 or 
            (composite_score > 0.8 and anomaly_score > 0.6)):
            return 'critical'
        elif (security_score > 0.5 or 
              composite_score > 0.6):
            return 'high'
        elif (security_score > 0.3 or 
              composite_score > 0.4):
            return 'medium'
        else:
            return 'low'
    
    def _estimate_confidence(self, result: Dict[str, Any]) -> float:
        """Estimate confidence in the ML analysis."""
        relevance = result.get('ml_relevance_score', 0.5)
        security = result.get('security_risk_score', 0.0)
        
        # Higher relevance and clear security indicators increase confidence
        base_confidence = relevance * 0.6
        
        # Security findings boost confidence
        if security > 0.5:
            base_confidence += 0.3
        elif security > 0.3:
            base_confidence += 0.2
        
        # Number of intelligence indicators
        indicators = result.get('ml_intelligence_indicators', [])
        indicator_bonus = min(len(indicators) * 0.05, 0.2)
        
        confidence = base_confidence + indicator_bonus
        return min(confidence, 1.0)
    
    def _cluster_similar_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar results using K-means."""
        if len(results) < 3:
            return results
        
        try:
            # Extract features for clustering
            features = []
            for result in results:
                feature_vector = [
                    result.get('ml_relevance_score', 0.5),
                    result.get('security_risk_score', 0.0),
                    result.get('anomaly_score', 0.0),
                    result.get('technology_score', 0.0),
                    len(result.get('ml_intelligence_indicators', []))
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Determine optimal number of clusters
            n_clusters = min(max(2, len(results) // 3), 5)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)
            
            # Add cluster information
            for i, result in enumerate(results):
                result['ml_cluster_id'] = int(cluster_labels[i])
            
            # Sort by cluster and score within cluster
            results.sort(key=lambda x: (x['ml_cluster_id'], -x.get('composite_ml_score', 0)))
            
        except Exception as e:
            logger.debug(f"Clustering failed: {e}")
        
        return results
    
    def _rank_results_ml(self, results: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Final ML-based ranking of results."""
        if not results:
            return results
        
        def calculate_final_score(result):
            composite = result.get('composite_ml_score', 0.5)
            confidence = result.get('ml_confidence', 0.5)
            
            # Risk category multiplier
            risk_multipliers = {
                'critical': 1.5,
                'high': 1.3,
                'medium': 1.1,
                'low': 1.0
            }
            
            risk_category = result.get('ml_risk_category', 'low')
            risk_multiplier = risk_multipliers.get(risk_category, 1.0)
            
            # Context bonus (if available)
            context_bonus = 0.0
            if context:
                # Add context-specific scoring here
                pass
            
            final_score = (composite * 0.7 + confidence * 0.3) * risk_multiplier + context_bonus
            return final_score
        
        # Sort by final score
        ranked_results = sorted(results, key=calculate_final_score, reverse=True)
        
        # Add final ranking
        for i, result in enumerate(ranked_results):
            result['ml_final_rank'] = i + 1
            result['ml_final_score'] = calculate_final_score(result)
        
        return ranked_results