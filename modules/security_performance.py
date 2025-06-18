import logging
import time
import hashlib
import re
import json
import os
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import asyncio
from functools import wraps
import ipaddress
from urllib.parse import urlparse
import threading

logger = logging.getLogger(__name__)

# Security and performance tracking
SECURITY_LOGS = deque(maxlen=1000)  # Keep last 1000 security events
PERFORMANCE_METRICS = defaultdict(list)
RATE_LIMITS = defaultdict(lambda: {"count": 0, "reset_time": time.time() + 60})
USER_SESSIONS = {}
THREAT_INDICATORS = set()
BLOCKED_PATTERNS = set()

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.request_times = deque(maxlen=100)
        self.error_count = 0
        self.success_count = 0
        self.start_time = time.time()
        self.memory_usage = []
        
    def log_request(self, duration: float, success: bool = True):
        """Log request performance metrics."""
        self.request_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.request_times:
            return {"status": "no_data"}
        
        avg_response_time = sum(self.request_times) / len(self.request_times)
        uptime = time.time() - self.start_time
        total_requests = self.success_count + self.error_count
        
        return {
            "average_response_time": avg_response_time,
            "uptime_seconds": uptime,
            "total_requests": total_requests,
            "success_rate": self.success_count / total_requests if total_requests > 0 else 0,
            "error_rate": self.error_count / total_requests if total_requests > 0 else 0,
            "requests_per_minute": len(self.request_times) / min(uptime / 60, 1),
            "status": "healthy" if avg_response_time < 5.0 and self.success_count > self.error_count else "degraded"
        }

# Global performance monitor
perf_monitor = PerformanceMonitor()

# Security Classes
class SecurityScanner:
    """Comprehensive security scanning and validation."""
    
    def __init__(self):
        self.malicious_patterns = [
            # XSS patterns (enhanced)
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*src=',
            r'<object[^>]*data=',
            r'<embed[^>]*src=',
            r'eval\s*\(',
            r'document\.write\s*\(',
            r'innerHTML\s*=',
            
            # SQL injection patterns (enhanced)
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)',
            r'(\bUNION\b.*\bSELECT\b)',
            r'(\b(OR|AND)\b\s+\d+\s*=\s*\d+)',
            r'(\'\s*(OR|AND)\s+\'\w+\'\s*=\s*\'\w+)',
            r'(;|--|/\*|\*/)',
            r'(\bEXEC\b|\bEXECUTE\b)',
            r'(\bxp_cmdshell\b)',
            
            # Command injection patterns (enhanced)
            r'(;|\||\&\&)\s*(rm|del|format|net\s+user)',
            r'`[^`]*`',
            r'\$\([^)]*\)',
            r'(&&|\|\|)',
            r'(nc|netcat)\s+-[lep]',
            r'wget\s+http',
            r'curl\s+http',
            r'/bin/(bash|sh|zsh)',
            r'powershell\s+-',
            
            # Path traversal (enhanced)
            r'\.\.\/|\.\.\\',
            r'\/etc\/passwd',
            r'\/windows\/system32',
            r'%2e%2e%2f',
            r'%252e%252e%252f',
            r'..%c0%af',
            r'..%c1%9c',
            
            # Code injection (enhanced)
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'shell_exec\s*\(',
            r'passthru\s*\(',
            r'proc_open\s*\(',
            r'popen\s*\(',
            r'assert\s*\(',
            
            # Reverse shell indicators
            r'/dev/tcp/',
            r'bash\s+-i\s+>&',
            r'python.*socket.*connect',
            r'perl.*socket.*connect',
            r'ruby.*socket.*connect',
            r'nc.*-e.*/',
            
            # Malware/exploit patterns
            r'metasploit',
            r'meterpreter',
            r'shikata_ga_nai',
            r'alpha_mixed',
            r'x86/shikata_ga_nai',
            r'windows/meterpreter',
            r'linux/x86/shell',
            
            # Steganography indicators
            r'steghide',
            r'outguess',
            r'jsteg',
            r'f5.*steganography',
            r'lsb.*steganography',
            
            # Cryptocurrency mining
            r'stratum\+tcp://',
            r'cryptonight',
            r'monero.*mining',
            r'bitcoin.*mining',
            r'ethereum.*mining',
            
            # Ransomware indicators
            r'\.locked$',
            r'\.encrypted$',
            r'DECRYPT.*INSTRUCTION',
            r'ransom.*payment',
            r'bitcoin.*payment',
        ]
        
        self.suspicious_keywords = [
            'payload', 'exploit', 'reverse shell', 'backdoor', 'rootkit',
            'keylogger', 'ransomware', 'trojan', 'botnet', 'ddos',
            'vulnerability', 'zero-day', 'privilege escalation', 'buffer overflow',
            'shellcode', 'exploit kit', 'malware', 'spyware', 'adware',
            'phishing', 'social engineering', 'credential harvesting',
            'lateral movement', 'persistence', 'command and control', 'c2',
            'exfiltration', 'steganography', 'obfuscation', 'evasion',
            'cryptojacking', 'cryptocurrency mining', 'botnet herder',
            'rat', 'remote access tool', 'advanced persistent threat', 'apt'
        ]
        
        # File type validation patterns
        self.dangerous_extensions = {
            '.exe', '.scr', '.bat', '.cmd', '.com', '.pif', '.vbs', '.vbe',
            '.js', '.jse', '.jar', '.app', '.deb', '.pkg', '.rpm', '.dmg',
            '.msi', '.dll', '.sys', '.drv', '.ocx', '.cpl', '.scf', '.lnk',
            '.url', '.inf', '.reg', '.ps1', '.psm1', '.psd1', '.ps1xml',
            '.psc1', '.psc2', '.msh', '.msh1', '.msh2', '.mshxml', '.msh1xml',
            '.msh2xml', '.gadget', '.workflow', '.action'
        }
        
        # Network security patterns
        self.network_indicators = [
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IP addresses
            r'(?:https?://)?(?:[-\w.])+(?:\.[a-zA-Z]{2,3})+(?:/.*)?',  # URLs
            r'\b[A-Fa-f0-9]{32}\b',  # MD5 hashes
            r'\b[A-Fa-f0-9]{40}\b',  # SHA1 hashes
            r'\b[A-Fa-f0-9]{64}\b',  # SHA256 hashes
        ]
        
    def scan_input(self, content: str, context: str = "") -> Dict[str, Any]:
        """Scan input for security threats - optimized for cybersecurity education and security research."""
        threats = []
        severity = "low"
        risk_score = 0
        
        # Check if this is clearly educational/cybersecurity content
        educational_keywords = [
            'cybersecurity', 'security', 'penetration testing', 'ethical hacking',
            'vulnerability', 'analysis', 'tutorial', 'education', 'learning',
            'red team', 'blue team', 'bug bounty', 'assessment', 'research',
            'what is', 'how does', 'explain', 'describe', 'help me understand'
        ]
        
        is_educational = any(keyword in content.lower() for keyword in educational_keywords)
        
        # Skip most security checks for educational content
        if is_educational:
            return {
                "is_safe": True,
                "severity": "low",
                "risk_score": 0,
                "threats": [],
                "scan_time": datetime.now().isoformat(),
                "context": context,
                "educational_content": True,
                "message": "Educational/cybersecurity content detected - security filtering bypassed"
            }
        
        # Enhanced pattern matching with scoring
        for pattern in self.malicious_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                threat_severity = self._assess_pattern_severity(pattern, matches)
                threats.append({
                    "type": "malicious_pattern",
                    "pattern": pattern,
                    "matches": len(matches),
                    "severity": threat_severity,
                    "examples": [str(match)[:50] for match in matches[:3]],  # Show first 3 matches, truncated
                    "description": f"Detected {len(matches)} potential malicious pattern(s)",
                    "category": self._categorize_pattern(pattern)
                })
                risk_score += self._calculate_pattern_risk(pattern, len(matches))
                if threat_severity == "critical":
                    severity = "critical"
                elif threat_severity == "high" and severity != "critical":
                    severity = "high"
                elif threat_severity == "medium" and severity in ["low", "medium"]:
                    severity = "medium"
        
        # Enhanced keyword detection with context analysis
        keyword_matches = {}
        for keyword in self.suspicious_keywords:
            if keyword.lower() in content.lower():
                context_relevance = self._assess_keyword_context(keyword, content, context)
                keyword_matches[keyword] = context_relevance
                
                threat_severity = "high" if context_relevance > 0.7 else "medium" if context_relevance > 0.4 else "low"
                threats.append({
                    "type": "suspicious_keyword",
                    "keyword": keyword,
                    "severity": threat_severity,
                    "context_relevance": context_relevance,
                    "description": f"Suspicious keyword '{keyword}' detected (relevance: {context_relevance:.2f})",
                    "category": "keyword_analysis"
                })
                risk_score += context_relevance * 10
                
                if threat_severity == "high" and severity not in ["critical"]:
                    severity = "high"
                elif threat_severity == "medium" and severity in ["low", "medium"]:
                    severity = "medium"
        
        # File extension analysis
        dangerous_extensions = [ext for ext in self.dangerous_extensions if ext in content.lower()]
        if dangerous_extensions:
            threats.append({
                "type": "dangerous_file_extension",
                "extensions": dangerous_extensions,
                "severity": "high",
                "description": f"Dangerous file extensions detected: {', '.join(dangerous_extensions)}",
                "category": "file_security"
            })
            risk_score += len(dangerous_extensions) * 15
            if severity not in ["critical"]:
                severity = "high"
        
        # Network indicators analysis with enhanced detection
        network_matches = []
        for pattern in self.network_indicators:
            matches = re.findall(pattern, content)
            if matches:
                network_matches.extend(matches)
        
        if network_matches:
            suspicious_networks = self._analyze_network_indicators(network_matches)
            if suspicious_networks:
                threats.append({
                    "type": "suspicious_network_indicator",
                    "indicators": suspicious_networks,
                    "severity": "medium",
                    "description": f"Suspicious network indicators detected: {len(suspicious_networks)} indicators",
                    "category": "network_security"
                })
                risk_score += len(suspicious_networks) * 5
        
        # Content analysis for advanced threats
        advanced_threats = self._detect_advanced_threats(content)
        if advanced_threats:
            threats.extend(advanced_threats)
            risk_score += sum(threat.get('risk_score', 0) for threat in advanced_threats)
            
            critical_advanced = [t for t in advanced_threats if t.get('severity') == 'critical']
            high_advanced = [t for t in advanced_threats if t.get('severity') == 'high']
            
            if critical_advanced:
                severity = "critical"
            elif high_advanced and severity != "critical":
                severity = "high"
        
        # Length and encoding checks (enhanced)
        if len(content) > 100000:  # 100KB threshold
            threats.append({
                "type": "length_check",
                "severity": "medium",
                "length": len(content),
                "description": f"Unusually long input detected: {len(content):,} characters",
                "category": "input_validation"
            })
            risk_score += 8
        
        # Enhanced encoding detection
        encoding_threats = self._check_advanced_encoding(content)
        if encoding_threats:
            threats.extend(encoding_threats)
            risk_score += sum(threat.get('risk_score', 0) for threat in encoding_threats)
        
        # Real-time threat intelligence integration
        intel_threats = self._check_threat_intelligence(content)
        if intel_threats:
            threats.extend(intel_threats)
            risk_score += sum(threat.get('risk_score', 0) for threat in intel_threats)
        
        # Final risk assessment with context consideration
        final_severity = self._calculate_final_severity(risk_score, severity, context)
        
        result = {
            "is_safe": len([t for t in threats if t.get('severity') in ['high', 'critical']]) == 0,
            "severity": final_severity,
            "risk_score": min(risk_score, 100),  # Cap at 100
            "threats": threats,
            "scan_time": datetime.now().isoformat(),
            "context": context,
            "total_patterns_checked": len(self.malicious_patterns),
            "total_keywords_checked": len(self.suspicious_keywords),
            "threat_categories": list(set(t.get('category', 'unknown') for t in threats))
        }
        
        # Enhanced logging with threat categorization
        if threats:
            high_severity_threats = [t for t in threats if t.get('severity') in ['high', 'critical']]
            if high_severity_threats:
                logger.warning(f"🚨 High-severity threats detected: {len(high_severity_threats)} threats")
            self._log_security_event("threat_detected", result)
        
        return result
    
    def _assess_pattern_severity(self, pattern: str, matches: List[str]) -> str:
        """Assess the severity of a detected pattern."""
        # Critical patterns
        critical_indicators = [
            'meterpreter', 'metasploit', 'reverse shell', '/dev/tcp/',
            'xp_cmdshell', 'powershell -', 'DECRYPT.*INSTRUCTION'
        ]
        
        # High severity patterns
        high_indicators = [
            'script', 'eval', 'exec', 'UNION.*SELECT', 'nc.*-e',
            'wget http', 'curl http', 'shell_exec'
        ]
        
        pattern_lower = pattern.lower()
        
        if any(indicator in pattern_lower for indicator in critical_indicators):
            return "critical"
        elif any(indicator in pattern_lower for indicator in high_indicators):
            return "high"
        elif len(matches) > 5:  # Multiple matches increase severity
            return "high"
        elif len(matches) > 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_pattern_risk(self, pattern: str, match_count: int) -> float:
        """Calculate risk score for a pattern match."""
        base_score = 10
        
        # Critical patterns get higher base scores
        if any(indicator in pattern.lower() for indicator in ['meterpreter', 'metasploit', 'xp_cmdshell']):
            base_score = 25
        elif any(indicator in pattern.lower() for indicator in ['script', 'eval', 'exec', 'union']):
            base_score = 15
        
        # Multiple matches increase risk
        multiplier = min(match_count * 0.5 + 1, 3)  # Cap multiplier at 3
        
        return base_score * multiplier
    
    def _assess_keyword_context(self, keyword: str, content: str, context: str) -> float:
        """Assess how relevant a keyword is in its context - optimized for cybersecurity education."""
        # Cybersecurity/educational contexts that are completely legitimate
        legitimate_indicators = [
            'tutorial', 'guide', 'learn', 'education', 'example', 'demonstration',
            'training', 'course', 'academic', 'research', 'study', 'explanation',
            'cybersecurity', 'security', 'penetration testing', 'pen test', 'red team',
            'blue team', 'bug bounty', 'vulnerability', 'assessment', 'analysis',
            'cve', 'security research', 'defensive', 'ethical hacking', 'testing',
            'analysis', 'what is', 'how does', 'explain', 'describe', 'help'
        ]
        
        content_lower = content.lower()
        context_lower = context.lower()
        
        # Check for legitimate cybersecurity/educational context
        legitimate_score = sum(1 for indicator in legitimate_indicators 
                             if indicator in content_lower or indicator in context_lower)
        
        # If this is clearly educational/cybersecurity content, return very low relevance
        if legitimate_score > 0:
            return 0.1  # Very low threat relevance for educational content
        
        # Only flag as high relevance for clearly malicious contexts
        malicious_indicators = [
            'execute attack', 'deploy payload', 'compromise system', 'backdoor access',
            'steal credentials', 'extract data', 'unauthorized access', 'bypass security'
        ]
        
        malicious_score = sum(1 for indicator in malicious_indicators 
                            if indicator in content_lower or indicator in context_lower)
        
        if malicious_score > 0:
            return 0.8  # High relevance only for clearly malicious intent
        
        return 0.2  # Default low relevance for ambiguous content
    
    def _analyze_network_indicators(self, network_matches: List[str]) -> List[Dict[str, Any]]:
        """Analyze network indicators for suspicious activity."""
        suspicious = []
        
        for match in network_matches:
            # Check for private/internal IPs (less suspicious in educational context)
            if re.match(r'(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.)', match):
                continue
            
            # Check for suspicious TLDs
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.bit', '.onion']
            if any(tld in match for tld in suspicious_tlds):
                suspicious.append({
                    "indicator": match,
                    "type": "suspicious_tld",
                    "risk": "medium"
                })
            
            # Check for URL shorteners
            url_shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly']
            if any(shortener in match for shortener in url_shorteners):
                suspicious.append({
                    "indicator": match,
                    "type": "url_shortener",
                    "risk": "low"
                })
        
        return suspicious
    
    def _detect_advanced_threats(self, content: str) -> List[Dict[str, Any]]:
        """Detect advanced persistent threats and sophisticated attacks."""
        threats = []
        
        # Obfuscation detection
        if self._detect_obfuscation(content):
            threats.append({
                "type": "obfuscation_detected",
                "severity": "high",
                "risk_score": 20,
                "description": "Content appears to be obfuscated or encoded to evade detection"
            })
        
        # Polyglot file indicators
        polyglot_indicators = [b'\x89PNG', b'\xff\xd8\xff', b'GIF8', b'PK\x03\x04']
        content_bytes = content.encode('utf-8', errors='ignore')
        polyglot_count = sum(1 for indicator in polyglot_indicators if indicator in content_bytes)
        
        if polyglot_count > 1:
            threats.append({
                "type": "polyglot_file_indicator",
                "severity": "high",
                "risk_score": 15,
                "description": f"Multiple file format signatures detected ({polyglot_count})"
            })
        
        # Living-off-the-land techniques
        lol_techniques = [
            'powershell.*iex', 'powershell.*invoke-expression',
            'certutil.*-decode', 'certutil.*-urlcache',
            'bitsadmin.*transfer', 'regsvr32.*scrobj.dll',
            'rundll32.*javascript', 'mshta.*http',
            'wmic.*process.*call.*create'
        ]
        
        for technique in lol_techniques:
            if re.search(technique, content, re.IGNORECASE):
                threats.append({
                    "type": "living_off_the_land",
                    "technique": technique,
                    "severity": "high",
                    "risk_score": 25,
                    "description": "Living-off-the-land technique detected"
                })
        
        return threats
    
    def _detect_obfuscation(self, content: str) -> bool:
        """Detect if content is obfuscated."""
        # High entropy check
        if len(set(content)) / len(content) > 0.8 and len(content) > 100:
            return True
        
        # Base64 with high frequency
        base64_pattern = r'[A-Za-z0-9+/]{50,}={0,2}'
        base64_matches = len(re.findall(base64_pattern, content))
        if base64_matches > 5:
            return True
        
        # Excessive escaping
        escape_count = content.count('\\x') + content.count('%')
        if escape_count > len(content) * 0.1:
            return True
        
        return False
    
    def _check_advanced_encoding(self, content: str) -> List[Dict[str, Any]]:
        """Enhanced encoding detection with multiple techniques."""
        threats = []
        
        # Base64 detection with payload analysis
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        base64_matches = re.findall(base64_pattern, content)
        
        if base64_matches:
            suspicious_b64 = []
            for b64_str in base64_matches[:10]:  # Check first 10 matches
                try:
                    import base64
                    decoded = base64.b64decode(b64_str).decode('utf-8', errors='ignore')
                    
                    # Check decoded content for threats
                    if any(pattern in decoded.lower() for pattern in 
                          ['script', 'eval', 'exec', 'powershell', 'cmd', '/bin/']):
                        suspicious_b64.append({
                            "encoded": b64_str[:50] + "..." if len(b64_str) > 50 else b64_str,
                            "decoded_snippet": decoded[:100] + "..." if len(decoded) > 100 else decoded
                        })
                except:
                    continue
            
            if suspicious_b64:
                threats.append({
                    "type": "malicious_base64",
                    "severity": "high",
                    "risk_score": 20,
                    "suspicious_payloads": len(suspicious_b64),
                    "examples": suspicious_b64[:3],
                    "description": f"Suspicious Base64 encoded content detected: {len(suspicious_b64)} payloads"
                })
        
        # Hex encoding detection
        hex_pattern = r'(?:\\x[0-9A-Fa-f]{2}){10,}'
        hex_matches = re.findall(hex_pattern, content)
        if hex_matches:
            threats.append({
                "type": "hex_encoding",
                "severity": "medium",
                "risk_score": 10,
                "matches": len(hex_matches),
                "description": f"Hex encoded content detected: {len(hex_matches)} sequences"
            })
        
        # URL encoding with high frequency
        url_encoded_pattern = r'%[0-9A-Fa-f]{2}'
        url_encoded_count = len(re.findall(url_encoded_pattern, content))
        if url_encoded_count > 20:
            threats.append({
                "type": "excessive_url_encoding",
                "severity": "medium",
                "risk_score": 8,
                "count": url_encoded_count,
                "description": f"Excessive URL encoding detected: {url_encoded_count} encoded characters"
            })
        
        return threats
    
    def _categorize_pattern(self, pattern: str) -> str:
        """Categorize security patterns by type."""
        pattern_lower = pattern.lower()
        
        if any(indicator in pattern_lower for indicator in ['script', 'iframe', 'object', 'embed', 'javascript']):
            return "xss_injection"
        elif any(indicator in pattern_lower for indicator in ['select', 'union', 'insert', 'delete', 'drop']):
            return "sql_injection"
        elif any(indicator in pattern_lower for indicator in ['cmd', 'exec', 'system', 'shell']):
            return "command_injection"
        elif any(indicator in pattern_lower for indicator in [r'\.\.', 'etc/passwd', 'windows/system32']):
            return "path_traversal"
        elif any(indicator in pattern_lower for indicator in ['metasploit', 'meterpreter', 'exploit']):
            return "exploit_framework"
        elif any(indicator in pattern_lower for indicator in ['steganography', 'obfuscation']):
            return "evasion_technique"
        else:
            return "general_malicious"
    
    def _check_threat_intelligence(self, content: str) -> List[Dict[str, Any]]:
        """Check content against real-time threat intelligence indicators."""
        threats = []
        
        # Known malicious domains (simplified threat intel)
        malicious_domains = [
            'malware-traffic-analysis.net',
            'exploit-db.com',
            'rapid7.com/db',
            # Add more known threat intel domains
        ]
        
        # Check for references to known malicious domains
        for domain in malicious_domains:
            if domain in content.lower():
                threats.append({
                    "type": "threat_intelligence_match",
                    "domain": domain,
                    "severity": "medium",
                    "risk_score": 15,
                    "description": f"Reference to known security research domain: {domain}",
                    "category": "threat_intelligence"
                })
        
        # Check for hash patterns that might be IOCs
        hash_patterns = {
            'md5': r'\b[a-fA-F0-9]{32}\b',
            'sha1': r'\b[a-fA-F0-9]{40}\b',
            'sha256': r'\b[a-fA-F0-9]{64}\b'
        }
        
        for hash_type, pattern in hash_patterns.items():
            matches = re.findall(pattern, content)
            if len(matches) > 3:  # Multiple hashes might indicate IOC sharing
                threats.append({
                    "type": "multiple_hash_indicators",
                    "hash_type": hash_type,
                    "count": len(matches),
                    "severity": "low",
                    "risk_score": 5,
                    "description": f"Multiple {hash_type} hashes detected: {len(matches)} hashes",
                    "category": "indicator_analysis"
                })
        
        return threats
    
    def _calculate_final_severity(self, risk_score: float, initial_severity: str, context: str) -> str:
        """Calculate final severity based on risk score, initial assessment, and context."""
        # Context-based adjustments
        educational_context = any(word in context.lower() for word in [
            'tutorial', 'guide', 'learn', 'education', 'example', 'demo'
        ])
        
        research_context = any(word in context.lower() for word in [
            'research', 'analysis', 'study', 'academic', 'paper'
        ])
        
        # Adjust risk score based on context
        adjusted_score = risk_score
        if educational_context:
            adjusted_score *= 0.7  # Reduce severity for educational content
        elif research_context:
            adjusted_score *= 0.8  # Slight reduction for research content
        
        # Final severity calculation
        if adjusted_score >= 80:
            return "critical"
        elif adjusted_score >= 50:
            return "high"
        elif adjusted_score >= 25:
            return "medium"
        elif adjusted_score >= 10:
            return "low"
        else:
            return initial_severity
    
    def scan_file_upload(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Comprehensive file upload security scanning."""
        threats = []
        risk_score = 0
        
        try:
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # File extension check
            if file_ext in self.dangerous_extensions:
                threats.append({
                    "type": "dangerous_extension",
                    "extension": file_ext,
                    "severity": "critical",
                    "description": f"Dangerous file extension: {file_ext}"
                })
                risk_score += 30
            
            # File size anomalies
            if file_size > 500 * 1024 * 1024:  # > 500MB
                threats.append({
                    "type": "large_file",
                    "size": file_size,
                    "severity": "medium",
                    "description": f"Large file size: {file_size} bytes"
                })
                risk_score += 10
            elif file_size == 0:
                threats.append({
                    "type": "empty_file",
                    "severity": "low",
                    "description": "Empty file detected"
                })
                risk_score += 5
            
            # File content analysis
            with open(file_path, 'rb') as f:
                file_header = f.read(512)
                
            # Magic byte analysis
            magic_threats = self._analyze_magic_bytes(file_header, file_ext)
            threats.extend(magic_threats)
            risk_score += sum(threat.get('risk_score', 0) for threat in magic_threats)
            
            # Entropy analysis
            entropy = self._calculate_entropy(file_header)
            if entropy > 7.5:  # High entropy indicates encryption/compression/randomness
                threats.append({
                    "type": "high_entropy",
                    "entropy": entropy,
                    "severity": "medium",
                    "description": f"High entropy detected: {entropy:.2f} (possible encryption/obfuscation)"
                })
                risk_score += 8
            
        except Exception as e:
            threats.append({
                "type": "analysis_error",
                "error": str(e),
                "severity": "medium",
                "description": f"Error during file analysis: {e}"
            })
            risk_score += 5
        
        severity = self._calculate_final_severity(risk_score, "low", "")
        
        return {
            "is_safe": len(threats) == 0,
            "severity": severity,
            "risk_score": min(risk_score, 100),
            "threats": threats,
            "file_info": {
                "filename": filename,
                "extension": file_ext,
                "size": file_size
            },
            "scan_time": datetime.now().isoformat()
        }
    
    def _analyze_magic_bytes(self, file_header: bytes, expected_ext: str) -> List[Dict[str, Any]]:
        """Analyze file magic bytes for inconsistencies."""
        threats = []
        
        # Common file signatures
        signatures = {
            b'\x89PNG\r\n\x1a\n': ['.png'],
            b'\xff\xd8\xff': ['.jpg', '.jpeg'],
            b'GIF8': ['.gif'],
            b'RIFF': ['.wav', '.avi'],
            b'ftyp': ['.mp4', '.mov'],
            b'PK\x03\x04': ['.zip', '.docx', '.xlsx', '.jar'],
            b'\x7fELF': ['.elf', ''],  # Linux executable
            b'MZ': ['.exe', '.dll'],  # Windows executable
            b'\xcf\xfa\xed\xfe': [''],  # Mach-O (macOS executable)
        }
        
        detected_type = None
        for signature, extensions in signatures.items():
            if file_header.startswith(signature):
                detected_type = extensions
                break
        
        if detected_type:
            if expected_ext not in detected_type and expected_ext != '':
                threats.append({
                    "type": "file_type_mismatch",
                    "expected": expected_ext,
                    "detected": detected_type,
                    "severity": "high",
                    "risk_score": 20,
                    "description": f"File extension mismatch: {expected_ext} vs {detected_type}"
                })
            
            # Check for executable disguised as media
            if any(ext in detected_type for ext in ['', '.exe', '.dll', '.elf']) and expected_ext in ['.jpg', '.png', '.mp4', '.mp3']:
                threats.append({
                    "type": "executable_disguised",
                    "severity": "critical",
                    "risk_score": 35,
                    "description": "Executable file disguised as media file"
                })
        
        return threats
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0
        
        # Count byte frequencies
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0
        data_len = len(data)
        for count in freq.values():
            p = count / data_len
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        SECURITY_LOGS.append(event)
        logger.warning(f"Security event: {event_type} - {details.get('severity', 'unknown')}")

class RateLimiter:
    """Advanced rate limiting with multiple strategies."""
    
    def __init__(self):
        self.limits = {
            "requests_per_minute": 30,
            "requests_per_hour": 500,
            "image_analysis_per_hour": 50,
            "url_processing_per_hour": 100,
            "file_uploads_per_hour": 20
        }
        
    def check_rate_limit(self, user_id: str, action_type: str = "request") -> Tuple[bool, str]:
        """Check if action is within rate limits."""
        current_time = time.time()
        
        # Clean old entries
        if user_id not in USER_SESSIONS:
            USER_SESSIONS[user_id] = defaultdict(list)
        
        user_actions = USER_SESSIONS[user_id][action_type]
        
        # Remove entries older than 1 hour
        USER_SESSIONS[user_id][action_type] = [
            timestamp for timestamp in user_actions 
            if current_time - timestamp < 3600
        ]
        
        # Check hourly limit
        hourly_limit = self.limits.get(f"{action_type}_per_hour", 100)
        if len(USER_SESSIONS[user_id][action_type]) >= hourly_limit:
            return False, f"Hourly limit exceeded for {action_type} ({hourly_limit}/hour)"
        
        # Check minute limit for general requests
        if action_type == "request":
            minute_actions = [
                timestamp for timestamp in USER_SESSIONS[user_id][action_type]
                if current_time - timestamp < 60
            ]
            minute_limit = self.limits["requests_per_minute"]
            if len(minute_actions) >= minute_limit:
                return False, f"Rate limit exceeded ({minute_limit}/minute)"
        
        # Record the action
        USER_SESSIONS[user_id][action_type].append(current_time)
        return True, "OK"

class ContextAnalyzer:
    """Intelligent context analysis for better responses."""
    
    def __init__(self):
        self.conversation_patterns = {
            "technical_question": [
                r'\bhow\s+(does|do|can|to)\b',
                r'\bwhat\s+is\b',
                r'\bexplain\b',
                r'\btutorial\b',
                r'\bguide\b'
            ],
            "security_analysis": [
                r'\banalyze\b',
                r'\bscan\b',
                r'\bvulnerability\b',
                r'\bsecurity\b',
                r'\bthreat\b'
            ],
            "code_request": [
                r'\bcode\b',
                r'\bscript\b',
                r'\bexample\b',
                r'\bsample\b',
                r'\bpayload\b'
            ],
            "troubleshooting": [
                r'\berror\b',
                r'\bproblem\b',
                r'\bissue\b',
                r'\bnot\s+working\b',
                r'\bfix\b'
            ]
        }
        
        self.security_contexts = {
            "penetration_testing": [
                'pentest', 'penetration', 'vulnerability assessment', 'security testing'
            ],
            "incident_response": [
                'incident', 'breach', 'attack', 'compromise', 'forensics'
            ],
            "malware_analysis": [
                'malware', 'virus', 'trojan', 'ransomware', 'analysis'
            ],
            "network_security": [
                'network', 'firewall', 'intrusion', 'monitoring', 'traffic'
            ]
        }
    
    def analyze_context(self, message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyze message context for intelligent responses."""
        context = {
            "intent": self._detect_intent(message),
            "security_domain": self._detect_security_domain(message),
            "complexity_level": self._assess_complexity(message),
            "requires_caution": self._requires_caution(message),
            "suggested_followups": self._generate_followups(message),
            "response_style": self._determine_response_style(message)
        }
        
        # Analyze conversation flow
        if conversation_history:
            context["conversation_flow"] = self._analyze_conversation_flow(conversation_history)
        
        return context
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message."""
        for intent, patterns in self.conversation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return intent
        return "general_inquiry"
    
    def _detect_security_domain(self, message: str) -> Optional[str]:
        """Detect cybersecurity domain context."""
        for domain, keywords in self.security_contexts.items():
            if any(keyword.lower() in message.lower() for keyword in keywords):
                return domain
        return None
    
    def _assess_complexity(self, message: str) -> str:
        """Assess the complexity level needed for response."""
        technical_indicators = len(re.findall(r'\b(protocol|algorithm|encryption|authentication|vulnerability|exploit)\b', message, re.IGNORECASE))
        
        if technical_indicators >= 3:
            return "advanced"
        elif technical_indicators >= 1:
            return "intermediate"
        else:
            return "beginner"
    
    def _requires_caution(self, message: str) -> bool:
        """Check if response requires security cautions."""
        caution_keywords = [
            'exploit', 'hack', 'attack', 'penetration', 'vulnerability',
            'malware', 'backdoor', 'shell', 'payload'
        ]
        return any(keyword in message.lower() for keyword in caution_keywords)
    
    def _generate_followups(self, message: str) -> List[str]:
        """Generate suggested follow-up questions."""
        followups = []
        
        if 'vulnerability' in message.lower():
            followups.extend([
                "Would you like to know about vulnerability assessment tools?",
                "Are you interested in remediation strategies?",
                "Do you need information about vulnerability disclosure?"
            ])
        
        if any(word in message.lower() for word in ['pentest', 'penetration']):
            followups.extend([
                "Would you like to learn about different penetration testing methodologies?",
                "Are you interested in specific tools and techniques?",
                "Do you need guidance on reporting findings?"
            ])
        
        return followups[:3]  # Limit to 3 suggestions
    
    def _determine_response_style(self, message: str) -> str:
        """Determine appropriate response style."""
        if any(word in message.lower() for word in ['urgent', 'emergency', 'breach', 'attack']):
            return "urgent_concise"
        elif 'tutorial' in message.lower() or 'guide' in message.lower():
            return "educational_detailed"
        elif 'quick' in message.lower() or 'fast' in message.lower():
            return "concise_practical"
        else:
            return "balanced_comprehensive"
    
    def _analyze_conversation_flow(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze conversation flow patterns."""
        if len(history) < 2:
            return {"pattern": "initial", "continuity": "new_topic"}
        
        recent_topics = []
        for item in history[-3:]:
            if 'user' in item:
                recent_topics.extend(re.findall(r'\b(security|network|malware|vulnerability|pentest|hack)\b', 
                                              item['user'], re.IGNORECASE))
        
        if len(set(recent_topics)) == 1:
            return {"pattern": "focused", "continuity": "same_topic", "topic": recent_topics[0].lower()}
        elif len(recent_topics) > 3:
            return {"pattern": "exploratory", "continuity": "related_topics"}
        else:
            return {"pattern": "mixed", "continuity": "varied_topics"}

# Security and performance decorators
def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            perf_monitor.log_request(duration, True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            perf_monitor.log_request(duration, False)
            logger.error(f"Performance monitor caught error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def security_scan(func):
    """Decorator to scan inputs for security threats."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        scanner = SecurityScanner()
        
        # Scan string arguments
        for arg in args:
            if isinstance(arg, str) and len(arg) > 10:
                scan_result = scanner.scan_input(arg, func.__name__)
                if not scan_result["is_safe"] and scan_result["severity"] == "high":
                    logger.warning(f"High severity threat detected in {func.__name__}")
                    return "⚠️ Security concern detected. Please review your input and try again."
        
        return await func(*args, **kwargs)
    return wrapper

def rate_limit(action_type: str = "request"):
    """Decorator to apply rate limiting."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get user ID from various sources
            user_id = "anonymous"
            
            # Look for user session in chainlit
            try:
                import chainlit as cl
                if hasattr(cl, 'user_session') and cl.user_session.get("user_id"):
                    user_id = cl.user_session.get("user_id")
            except:
                pass
            
            limiter = RateLimiter()
            allowed, message = limiter.check_rate_limit(user_id, action_type)
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for user {user_id}: {message}")
                return f"⏱️ {message}. Please wait before trying again."
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Utility functions
def get_security_status() -> Dict[str, Any]:
    """Get overall security status."""
    recent_events = [event for event in SECURITY_LOGS 
                    if datetime.fromisoformat(event["timestamp"]) > datetime.now() - timedelta(hours=1)]
    
    high_severity_events = [event for event in recent_events 
                          if event.get("details", {}).get("severity") == "high"]
    
    return {
        "status": "alert" if high_severity_events else "normal",
        "recent_events": len(recent_events),
        "high_severity_events": len(high_severity_events),
        "last_event": SECURITY_LOGS[-1]["timestamp"] if SECURITY_LOGS else None,
        "total_events": len(SECURITY_LOGS)
    }

def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report."""
    stats = perf_monitor.get_stats()
    security = get_security_status()
    
    return {
        "performance": stats,
        "security": security,
        "timestamp": datetime.now().isoformat(),
        "uptime": stats.get("uptime_seconds", 0),
        "health_score": calculate_health_score(stats, security)
    }

def calculate_health_score(perf_stats: Dict, security_stats: Dict) -> int:
    """Calculate overall system health score (0-100)."""
    score = 100
    
    # Performance penalties
    if perf_stats.get("average_response_time", 0) > 5:
        score -= 20
    if perf_stats.get("error_rate", 0) > 0.1:
        score -= 30
    
    # Security penalties
    if security_stats["status"] == "alert":
        score -= 40
    if security_stats["high_severity_events"] > 0:
        score -= 20
    
    return max(0, score)

# Context-aware response enhancement
def enhance_response_with_context(response: str, context: Dict[str, Any]) -> str:
    """Enhance response based on context analysis."""
    enhanced = response
    
    # Add complexity-appropriate explanations
    complexity = context.get("complexity_level", "intermediate")
    if complexity == "beginner":
        enhanced += "\n\n💡 **Beginner Tip:** This is an advanced security topic. Consider starting with basic security fundamentals if you're new to cybersecurity."
    elif complexity == "advanced":
        enhanced += "\n\n🎓 **Advanced Topic:** This response covers sophisticated security concepts. Ensure you have proper authorization and expertise before implementation."
    
    # Add follow-up suggestions
    followups = context.get("suggested_followups", [])
    if followups:
        enhanced += f"\n\n**🤔 Related Questions:**\n" + "\n".join([f"• {followup}" for followup in followups])
    
    return enhanced 