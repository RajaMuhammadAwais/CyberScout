"""
Advanced AI Engine for OSINT Reconnaissance
Implements state-of-the-art ML algorithms for enhanced accuracy and intelligence
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import re
import json
from datetime import datetime
import asyncio

# Core ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
import nltk

# Advanced NLP imports (optional)
SPACY_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try importing optional advanced libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

class AdvancedAIEngine:
    """Advanced AI engine with multiple ML algorithms for OSINT analysis."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize models
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.relevance_classifier = LogisticRegression(random_state=42)
        self.clustering_model = None
        
        # NLP models
        self.nlp_model = None
        self.sentence_transformer = None
        
        # Knowledge bases
        self.security_patterns = self._load_security_patterns()
        self.industry_keywords = self._load_industry_keywords()
        self.threat_indicators = self._load_threat_indicators()
        
        # Initialize advanced models if available
        self._initialize_advanced_models()
        
    def _initialize_advanced_models(self):
        """Initialize advanced NLP models if available."""
        try:
            if SPACY_AVAILABLE:
                # Try to load spaCy model
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                    logger.info("SpaCy model loaded successfully")
                except OSError:
                    logger.debug("SpaCy English model not found, using basic NLP")
            
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Load sentence transformer for semantic similarity
                try:
                    self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Sentence transformer loaded successfully")
                except Exception:
                    logger.debug("Sentence transformer failed to load, using basic similarity")
                
        except Exception as e:
            logger.debug(f"Advanced NLP models initialization failed: {e}")
    
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load security-related patterns for detection."""
        return {
            'vulnerabilities': [
                'sql injection', 'xss', 'csrf', 'rce', 'lfi', 'rfi',
                'buffer overflow', 'privilege escalation', 'directory traversal',
                'authentication bypass', 'authorization bypass', 'code injection'
            ],
            'sensitive_files': [
                'config', 'configuration', 'backup', 'dump', 'log', 'debug',
                'admin', 'password', 'secret', 'key', 'token', 'credential'
            ],
            'exposures': [
                'exposed', 'leaked', 'public', 'misconfigured', 'unsecured',
                'default password', 'weak password', 'no authentication'
            ],
            'technologies': [
                'wordpress', 'drupal', 'joomla', 'apache', 'nginx', 'mysql',
                'postgresql', 'mongodb', 'redis', 'elasticsearch', 'jenkins'
            ]
        }
    
    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load industry-specific keywords for context analysis."""
        return {
            'finance': [
                'bank', 'financial', 'payment', 'transaction', 'credit', 'loan',
                'investment', 'trading', 'insurance', 'fintech', 'blockchain'
            ],
            'healthcare': [
                'medical', 'health', 'patient', 'hospital', 'clinic', 'pharma',
                'healthcare', 'diagnosis', 'treatment', 'medicine', 'doctor'
            ],
            'education': [
                'university', 'college', 'school', 'student', 'academic',
                'research', 'education', 'learning', 'course', 'degree'
            ],
            'government': [
                'government', 'federal', 'state', 'local', 'public', 'agency',
                'department', 'official', 'citizen', 'municipal', 'county'
            ],
            'technology': [
                'software', 'tech', 'startup', 'saas', 'cloud', 'ai', 'ml',
                'data', 'analytics', 'development', 'platform', 'api'
            ],
            'ecommerce': [
                'shop', 'store', 'retail', 'ecommerce', 'marketplace', 'cart',
                'order', 'customer', 'product', 'inventory', 'shipping'
            ]
        }
    
    def _load_threat_indicators(self) -> Dict[str, float]:
        """Load threat indicators with risk scores."""
        return {
            'password': 0.9,
            'secret': 0.9,
            'api_key': 0.8,
            'token': 0.8,
            'credential': 0.8,
            'backup': 0.7,
            'dump': 0.7,
            'config': 0.6,
            'admin': 0.6,
            'debug': 0.5,
            'test': 0.4,
            'staging': 0.4,
            'dev': 0.3
        }
    
    async def analyze_target_context(self, target: str, target_type: str) -> Dict[str, Any]:
        """Perform advanced context analysis of the target using multiple ML techniques."""
        analysis = {
            'target': target,
            'target_type': target_type,
            'industry_classification': {},
            'semantic_features': {},
            'risk_indicators': [],
            'technology_stack_predictions': [],
            'vulnerability_likelihood': 0.0,
            'confidence_score': 0.0
        }
        
        try:
            # Industry classification using ML
            analysis['industry_classification'] = self._classify_industry(target)
            
            # Semantic analysis
            if self.nlp_model:
                analysis['semantic_features'] = await self._extract_semantic_features(target)
            
            # Risk assessment
            analysis['risk_indicators'] = self._identify_risk_indicators(target)
            
            # Technology prediction
            analysis['technology_stack_predictions'] = self._predict_technology_stack(target)
            
            # Vulnerability likelihood
            analysis['vulnerability_likelihood'] = self._calculate_vulnerability_likelihood(analysis)
            
            # Overall confidence
            analysis['confidence_score'] = self._calculate_confidence_score(analysis)
            
        except Exception as e:
            logger.error(f"Target context analysis failed: {e}")
        
        return analysis
    
    def _classify_industry(self, target: str) -> Dict[str, float]:
        """Classify target industry using keyword matching and ML."""
        target_lower = target.lower()
        industry_scores = {}
        
        for industry, keywords in self.industry_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in target_lower:
                    # Weight longer keywords more heavily
                    score += len(keyword) / len(target_lower)
            
            industry_scores[industry] = min(score, 1.0)
        
        # Normalize scores
        total_score = sum(industry_scores.values())
        if total_score > 0:
            industry_scores = {k: v/total_score for k, v in industry_scores.items()}
        
        return dict(sorted(industry_scores.items(), key=lambda x: x[1], reverse=True))
    
    async def _extract_semantic_features(self, target: str) -> Dict[str, Any]:
        """Extract semantic features using advanced NLP."""
        features = {
            'entities': [],
            'sentiment': 0.0,
            'topics': [],
            'complexity_score': 0.0
        }
        
        try:
            if self.nlp_model:
                doc = self.nlp_model(target)
                
                # Extract named entities
                features['entities'] = [
                    {'text': ent.text, 'label': ent.label_, 'confidence': 1.0}
                    for ent in doc.ents
                ]
                
                # Calculate complexity
                features['complexity_score'] = len(doc) / max(len(target.split()), 1)
            
            # Sentiment analysis
            blob = TextBlob(target)
            features['sentiment'] = blob.sentiment.polarity
            
        except Exception as e:
            logger.debug(f"Semantic feature extraction failed: {e}")
        
        return features
    
    def _identify_risk_indicators(self, target: str) -> List[Dict[str, Any]]:
        """Identify risk indicators in the target."""
        indicators = []
        target_lower = target.lower()
        
        for indicator, risk_score in self.threat_indicators.items():
            if indicator in target_lower:
                indicators.append({
                    'indicator': indicator,
                    'risk_score': risk_score,
                    'context': 'direct_match',
                    'position': target_lower.find(indicator)
                })
        
        # Pattern-based detection
        patterns = {
            'version_pattern': r'v?\d+\.\d+\.\d+',
            'api_pattern': r'api[_-]?v?\d*',
            'admin_pattern': r'admin|administrator|root',
            'test_pattern': r'test|testing|staging|dev'
        }
        
        for pattern_name, pattern in patterns.items():
            matches = re.finditer(pattern, target_lower, re.IGNORECASE)
            for match in matches:
                indicators.append({
                    'indicator': match.group(),
                    'risk_score': 0.3,
                    'context': pattern_name,
                    'position': match.start()
                })
        
        return sorted(indicators, key=lambda x: x['risk_score'], reverse=True)
    
    def _predict_technology_stack(self, target: str) -> List[Dict[str, float]]:
        """Predict likely technology stack based on target analysis."""
        predictions = []
        target_lower = target.lower()
        
        tech_indicators = {
            'wordpress': ['wp', 'wordpress', 'wp-content', 'wp-admin'],
            'drupal': ['drupal', 'node', 'sites/default'],
            'joomla': ['joomla', 'administrator', 'components'],
            'apache': ['apache', 'httpd'],
            'nginx': ['nginx'],
            'php': ['php', 'index.php'],
            'asp.net': ['asp', 'aspx', 'ashx'],
            'java': ['java', 'jsp', 'servlet'],
            'node.js': ['node', 'npm', 'js'],
            'python': ['python', 'py', 'django', 'flask'],
            'ruby': ['ruby', 'rails', 'rb']
        }
        
        for tech, indicators in tech_indicators.items():
            score = 0.0
            for indicator in indicators:
                if indicator in target_lower:
                    score += 0.2
            
            if score > 0:
                predictions.append({
                    'technology': tech,
                    'confidence': min(score, 1.0)
                })
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_vulnerability_likelihood(self, analysis: Dict[str, Any]) -> float:
        """Calculate likelihood of vulnerabilities based on analysis."""
        base_score = 0.1  # Base vulnerability likelihood
        
        # Risk indicators contribution
        risk_contribution = 0.0
        for indicator in analysis.get('risk_indicators', []):
            risk_contribution += indicator['risk_score'] * 0.1
        
        # Technology stack contribution
        tech_contribution = 0.0
        for tech in analysis.get('technology_stack_predictions', []):
            # Older or commonly vulnerable technologies
            if tech['technology'] in ['wordpress', 'drupal', 'joomla', 'php']:
                tech_contribution += tech['confidence'] * 0.2
        
        # Industry contribution
        industry_contribution = 0.0
        industries = analysis.get('industry_classification', {})
        high_target_industries = ['finance', 'government', 'healthcare']
        for industry in high_target_industries:
            if industry in industries:
                industry_contribution += industries[industry] * 0.15
        
        total_score = base_score + risk_contribution + tech_contribution + industry_contribution
        return min(total_score, 1.0)
    
    def _calculate_confidence_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis."""
        factors = []
        
        # Industry classification confidence
        industries = analysis.get('industry_classification', {})
        if industries:
            max_industry_score = max(industries.values())
            factors.append(max_industry_score)
        
        # Risk indicators confidence
        risk_indicators = analysis.get('risk_indicators', [])
        if risk_indicators:
            avg_risk_score = np.mean([r['risk_score'] for r in risk_indicators])
            factors.append(avg_risk_score)
        
        # Technology predictions confidence
        tech_predictions = analysis.get('technology_stack_predictions', [])
        if tech_predictions:
            avg_tech_confidence = np.mean([t['confidence'] for t in tech_predictions])
            factors.append(avg_tech_confidence)
        
        return np.mean(factors) if factors else 0.5
    
    async def generate_intelligent_queries(self, target: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent search queries using advanced ML techniques."""
        queries = []
        
        try:
            # Context-aware query generation
            industry_queries = self._generate_industry_specific_queries(target, context)
            queries.extend(industry_queries)
            
            # Technology-aware queries
            tech_queries = self._generate_technology_queries(target, context)
            queries.extend(tech_queries)
            
            # Risk-based queries
            risk_queries = self._generate_risk_based_queries(target, context)
            queries.extend(risk_queries)
            
            # Semantic queries using NLP
            if self.nlp_model:
                semantic_queries = await self._generate_semantic_queries(target, context)
                queries.extend(semantic_queries)
            
            # Advanced pattern queries
            pattern_queries = self._generate_pattern_based_queries(target, context)
            queries.extend(pattern_queries)
            
            # Rank and filter queries
            queries = self._rank_and_filter_queries(queries, context)
            
        except Exception as e:
            logger.error(f"Intelligent query generation failed: {e}")
        
        return queries[:25]  # Return top 25 queries
    
    def _generate_industry_specific_queries(self, target: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries specific to detected industry."""
        queries = []
        industries = context.get('industry_classification', {})
        
        industry_query_templates = {
            'finance': [
                'site:{target} "account" OR "transaction" filetype:pdf',
                'site:{target} "banking" OR "payment" sensitive',
                'site:{target} "credit card" OR "financial data"'
            ],
            'healthcare': [
                'site:{target} "patient" OR "medical" filetype:pdf',
                'site:{target} "health record" OR "medical data"',
                'site:{target} "hipaa" OR "phi" compliance'
            ],
            'education': [
                'site:{target} "student" OR "grade" filetype:xls',
                'site:{target} "academic" OR "research" data',
                'site:{target} "enrollment" OR "registration"'
            ],
            'government': [
                'site:{target} "classified" OR "restricted" filetype:pdf',
                'site:{target} "citizen" OR "taxpayer" data',
                'site:{target} "security clearance" OR "confidential"'
            ]
        }
        
        for industry, score in list(industries.items())[:3]:  # Top 3 industries
            if industry in industry_query_templates and score > 0.1:
                for template in industry_query_templates[industry]:
                    query = template.format(target=target)
                    queries.append({
                        'query': query,
                        'category': f'industry_{industry}',
                        'priority': score * 0.8,
                        'risk_level': 'high' if industry in ['finance', 'healthcare', 'government'] else 'medium',
                        'confidence': score
                    })
        
        return queries
    
    def _generate_technology_queries(self, target: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries based on predicted technology stack."""
        queries = []
        tech_predictions = context.get('technology_stack_predictions', [])
        
        tech_query_templates = {
            'wordpress': [
                'site:{target} "wp-config.php" OR "wp-admin"',
                'site:{target} "wordpress" version filetype:txt',
                'site:{target} "wp-content/uploads" filetype:sql'
            ],
            'drupal': [
                'site:{target} "sites/default" OR "settings.php"',
                'site:{target} "drupal" version vulnerability',
                'site:{target} "/node/" OR "/admin/"'
            ],
            'apache': [
                'site:{target} "apache" version OR "httpd.conf"',
                'site:{target} "server-status" OR "server-info"',
                'site:{target} ".htaccess" filetype:txt'
            ],
            'nginx': [
                'site:{target} "nginx" configuration',
                'site:{target} "nginx.conf" filetype:txt',
                'site:{target} "nginx" version vulnerability'
            ]
        }
        
        for tech_info in tech_predictions[:5]:  # Top 5 technologies
            tech = tech_info['technology']
            confidence = tech_info['confidence']
            
            if tech in tech_query_templates and confidence > 0.2:
                for template in tech_query_templates[tech]:
                    query = template.format(target=target)
                    queries.append({
                        'query': query,
                        'category': f'technology_{tech}',
                        'priority': confidence * 0.7,
                        'risk_level': 'high',
                        'confidence': confidence
                    })
        
        return queries
    
    def _generate_risk_based_queries(self, target: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries based on identified risk indicators."""
        queries = []
        risk_indicators = context.get('risk_indicators', [])
        
        for indicator_info in risk_indicators[:10]:  # Top 10 risk indicators
            indicator = indicator_info['indicator']
            risk_score = indicator_info['risk_score']
            
            risk_templates = [
                f'site:{target} "{indicator}" filetype:log',
                f'site:{target} "{indicator}" exposed OR leaked',
                f'site:{target} "{indicator}" configuration',
                f'"{indicator}" {target} vulnerability'
            ]
            
            for template in risk_templates:
                queries.append({
                    'query': template,
                    'category': f'risk_{indicator}',
                    'priority': risk_score,
                    'risk_level': 'high' if risk_score > 0.7 else 'medium',
                    'confidence': risk_score
                })
        
        return queries
    
    async def _generate_semantic_queries(self, target: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate semantic queries using NLP."""
        queries = []
        
        try:
            if not self.nlp_model:
                return queries
            
            doc = self.nlp_model(target)
            
            # Extract key entities and concepts
            entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE']]
            
            for entity in entities[:3]:  # Top 3 entities
                semantic_templates = [
                    f'"{entity}" sensitive information filetype:pdf',
                    f'"{entity}" internal OR confidential',
                    f'"{entity}" database OR backup',
                    f'"{entity}" api OR credentials'
                ]
                
                for template in semantic_templates:
                    queries.append({
                        'query': template,
                        'category': 'semantic_entity',
                        'priority': 0.6,
                        'risk_level': 'medium',
                        'confidence': 0.7
                    })
        
        except Exception as e:
            logger.debug(f"Semantic query generation failed: {e}")
        
        return queries
    
    def _generate_pattern_based_queries(self, target: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries based on advanced patterns."""
        queries = []
        
        # Advanced patterns
        patterns = {
            'subdomain_discovery': [
                f'site:*.{target} -site:{target}',
                f'"{target}" subdomain OR hostname',
                f'inurl:{target} site:github.com'
            ],
            'file_discovery': [
                f'site:{target} filetype:sql OR filetype:db',
                f'site:{target} filetype:config OR filetype:ini',
                f'site:{target} filetype:log OR filetype:backup'
            ],
            'credential_hunting': [
                f'site:pastebin.com "{target}" password',
                f'site:github.com "{target}" api_key OR token',
                f'"{target}" credentials leaked'
            ],
            'vulnerability_research': [
                f'"{target}" vulnerability OR exploit',
                f'"{target}" security advisory',
                f'"{target}" CVE OR security bulletin'
            ]
        }
        
        for category, templates in patterns.items():
            for template in templates:
                queries.append({
                    'query': template,
                    'category': category,
                    'priority': 0.5,
                    'risk_level': 'medium',
                    'confidence': 0.6
                })
        
        return queries
    
    def _rank_and_filter_queries(self, queries: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank and filter queries using ML scoring."""
        if not queries:
            return queries
        
        # Calculate composite scores
        for query in queries:
            priority = query.get('priority', 0.5)
            confidence = query.get('confidence', 0.5)
            risk_multiplier = {'high': 1.2, 'medium': 1.0, 'low': 0.8}.get(query.get('risk_level', 'medium'), 1.0)
            
            # Composite scoring
            query['composite_score'] = (priority * 0.4 + confidence * 0.3) * risk_multiplier
            
            # Add diversity bonus
            category = query.get('category', '')
            category_count = sum(1 for q in queries if q.get('category') == category)
            if category_count <= 3:  # Encourage diversity
                query['composite_score'] *= 1.1
        
        # Sort by composite score and remove duplicates
        unique_queries = []
        seen_queries = set()
        
        for query in sorted(queries, key=lambda x: x['composite_score'], reverse=True):
            query_text = query['query'].lower()
            if query_text not in seen_queries:
                unique_queries.append(query)
                seen_queries.add(query_text)
        
        return unique_queries
    
    async def analyze_search_results(self, results: List[Dict[str, Any]], target: str, query_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze search results using advanced ML techniques."""
        if not results:
            return results
        
        try:
            # Extract features from results
            features = self._extract_result_features(results, target)
            
            # Anomaly detection
            anomaly_scores = self._detect_anomalies(features)
            
            # Relevance scoring
            relevance_scores = await self._calculate_advanced_relevance(results, target, query_context)
            
            # Risk assessment
            risk_scores = self._assess_result_risks(results, query_context)
            
            # Combine all scores
            enhanced_results = []
            for i, result in enumerate(results):
                enhanced_result = result.copy()
                enhanced_result.update({
                    'anomaly_score': anomaly_scores[i] if i < len(anomaly_scores) else 0.0,
                    'relevance_score': relevance_scores[i] if i < len(relevance_scores) else 0.5,
                    'risk_score': risk_scores[i] if i < len(risk_scores) else 0.3,
                    'ml_confidence': self._calculate_ml_confidence(anomaly_scores[i] if i < len(anomaly_scores) else 0.0,
                                                                 relevance_scores[i] if i < len(relevance_scores) else 0.5,
                                                                 risk_scores[i] if i < len(risk_scores) else 0.3)
                })
                enhanced_results.append(enhanced_result)
            
            # Cluster similar results
            clustered_results = self._cluster_results(enhanced_results)
            
            return clustered_results
            
        except Exception as e:
            logger.error(f"Advanced result analysis failed: {e}")
            return results
    
    def _extract_result_features(self, results: List[Dict[str, Any]], target: str) -> np.ndarray:
        """Extract numerical features from search results for ML analysis."""
        features = []
        
        for result in results:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            url = result.get('url', '')
            
            # Text-based features
            title_length = len(title)
            snippet_length = len(snippet)
            url_length = len(url)
            
            # Target relevance
            target_in_title = 1 if target.lower() in title.lower() else 0
            target_in_snippet = 1 if target.lower() in snippet.lower() else 0
            target_in_url = 1 if target.lower() in url.lower() else 0
            
            # Security indicators
            security_keywords = ['password', 'admin', 'config', 'backup', 'secret', 'key']
            security_count = sum(1 for keyword in security_keywords 
                               if keyword in (title + snippet + url).lower())
            
            # File type indicators
            file_extensions = ['.pdf', '.doc', '.xls', '.sql', '.config', '.log']
            file_type_count = sum(1 for ext in file_extensions if ext in url.lower())
            
            feature_vector = [
                title_length, snippet_length, url_length,
                target_in_title, target_in_snippet, target_in_url,
                security_count, file_type_count
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _detect_anomalies(self, features: np.ndarray) -> List[float]:
        """Detect anomalous results that might be interesting."""
        if len(features) < 2:
            return [0.5] * len(features)
        
        try:
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
            
            # Convert to probability scores (anomalies get higher scores)
            prob_scores = [(1.0 if score == -1 else 0.3) for score in anomaly_scores]
            
            return prob_scores
            
        except Exception as e:
            logger.debug(f"Anomaly detection failed: {e}")
            return [0.5] * len(features)
    
    async def _calculate_advanced_relevance(self, results: List[Dict[str, Any]], target: str, context: Dict[str, Any]) -> List[float]:
        """Calculate advanced relevance scores using multiple techniques."""
        relevance_scores = []
        
        # Extract text content
        texts = []
        for result in results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            texts.append(text)
        
        if not texts:
            return [0.5] * len(results)
        
        try:
            # TF-IDF similarity
            tfidf_scores = self._calculate_tfidf_similarity(texts, target)
            
            # Semantic similarity (if available)
            semantic_scores = []
            if self.sentence_transformer:
                semantic_scores = await self._calculate_semantic_similarity(texts, target)
            else:
                semantic_scores = [0.5] * len(texts)
            
            # Combine scores
            for i in range(len(results)):
                tfidf_score = tfidf_scores[i] if i < len(tfidf_scores) else 0.3
                semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0.5
                
                # Weighted combination
                combined_score = tfidf_score * 0.6 + semantic_score * 0.4
                relevance_scores.append(combined_score)
            
        except Exception as e:
            logger.debug(f"Advanced relevance calculation failed: {e}")
            relevance_scores = [0.5] * len(results)
        
        return relevance_scores
    
    def _calculate_tfidf_similarity(self, texts: List[str], target: str) -> List[float]:
        """Calculate TF-IDF similarity scores."""
        try:
            # Include target in corpus for comparison
            corpus = texts + [target]
            
            # Fit TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Calculate similarities with target (last item)
            target_vector = tfidf_matrix[-1]
            similarities = cosine_similarity(tfidf_matrix[:-1], target_vector).flatten()
            
            return similarities.tolist()
            
        except Exception as e:
            logger.debug(f"TF-IDF similarity calculation failed: {e}")
            return [0.5] * len(texts)
    
    async def _calculate_semantic_similarity(self, texts: List[str], target: str) -> List[float]:
        """Calculate semantic similarity using sentence transformers."""
        try:
            if not self.sentence_transformer:
                return [0.5] * len(texts)
            
            # Encode texts and target
            text_embeddings = self.sentence_transformer.encode(texts)
            target_embedding = self.sentence_transformer.encode([target])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(text_embeddings, target_embedding).flatten()
            
            return similarities.tolist()
            
        except Exception as e:
            logger.debug(f"Semantic similarity calculation failed: {e}")
            return [0.5] * len(texts)
    
    def _assess_result_risks(self, results: List[Dict[str, Any]], context: Dict[str, Any]) -> List[float]:
        """Assess risk levels of search results."""
        risk_scores = []
        
        for result in results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            url = result.get('url', '').lower()
            content = f"{title} {snippet} {url}"
            
            risk_score = 0.0
            
            # Check for threat indicators
            for indicator, weight in self.threat_indicators.items():
                if indicator in content:
                    risk_score += weight * 0.2
            
            # Check for security patterns
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    if pattern in content:
                        risk_score += 0.15
            
            # File type risks
            high_risk_extensions = ['.sql', '.db', '.config', '.backup', '.log']
            for ext in high_risk_extensions:
                if ext in url:
                    risk_score += 0.3
            
            # URL structure risks
            if any(keyword in url for keyword in ['admin', 'private', 'internal']):
                risk_score += 0.2
            
            risk_scores.append(min(risk_score, 1.0))
        
        return risk_scores
    
    def _calculate_ml_confidence(self, anomaly_score: float, relevance_score: float, risk_score: float) -> float:
        """Calculate overall ML confidence score."""
        # Weighted combination of different scores
        confidence = (
            relevance_score * 0.4 +
            risk_score * 0.3 +
            anomaly_score * 0.3
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _cluster_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar results using ML."""
        if len(results) < 3:
            return results
        
        try:
            # Extract features for clustering
            features = []
            for result in results:
                feature_vector = [
                    result.get('relevance_score', 0.5),
                    result.get('risk_score', 0.3),
                    result.get('anomaly_score', 0.0),
                    len(result.get('title', '')),
                    len(result.get('url', ''))
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Perform clustering
            n_clusters = min(5, len(results) // 2)  # Reasonable number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)
            
            # Add cluster information to results
            for i, result in enumerate(results):
                result['cluster_id'] = int(cluster_labels[i])
                result['cluster_center_distance'] = float(
                    np.linalg.norm(features_array[i] - kmeans.cluster_centers_[cluster_labels[i]])
                )
            
            # Sort by cluster and relevance
            results.sort(key=lambda x: (x['cluster_id'], -x.get('relevance_score', 0)))
            
        except Exception as e:
            logger.debug(f"Result clustering failed: {e}")
        
        return results