import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

@dataclass
class CodeEvaluation:
    accuracy: float
    performance: float
    best_practices: float
    error_handling: float
    security: float
    scalability: float
    overall_score: float
    details: Dict[str, List[str]]

def evaluate_code_accuracy(code_content: str, streaming_system: str) -> Tuple[float, List[str]]:
    accuracy_checks = {
        'apache flink': [
            ('import org.apache.flink', 'Correct Flink imports'),
            ('StreamExecutionEnvironment', 'Uses StreamExecutionEnvironment'),
            ('DataStream', 'Uses DataStream API'),
            ('execute()', 'Has execution environment')
        ],
        'apache spark': [
            ('import org.apache.spark', 'Correct Spark imports'),
            ('SparkSession', 'Uses SparkSession'),
            ('sparkContext', 'Uses SparkContext'),
            ('createStructuredStream', 'Uses Structured Streaming')
        ],
        'apache storm': [
            ('import org.apache.storm', 'Correct Storm imports'),
            ('TopologyBuilder', 'Uses TopologyBuilder'),
            ('IBolt', 'Implements IBolt'),
            ('ISpout', 'Implements ISpout')
        ],
        'kafka stream': [
            ('import org.apache.kafka.streams', 'Correct Kafka imports'),
            ('KafkaStreams', 'Uses KafkaStreams'),
            ('StreamsBuilder', 'Uses StreamsBuilder'),
            ('Topology', 'Creates Topology')
        ]
    }

    system = streaming_system.lower()
    if system not in accuracy_checks:
        return 0.0, ['Unsupported streaming system']

    passed_checks = []
    failed_checks = []

    for pattern, description in accuracy_checks[system]:
        if pattern.lower() in code_content.lower():
            passed_checks.append(description)
        else:
            failed_checks.append(f"Missing: {description}")

    score = len(passed_checks) / len(accuracy_checks[system])
    return score, failed_checks

def evaluate_performance_patterns(code_content: str) -> Tuple[float, List[str]]:
    performance_patterns = [
        (r'\.setParallelism\(', 'Uses parallelism settings'),
        (r'\.repartition\(', 'Uses data repartitioning'),
        (r'\.cache\(|\.persist\(', 'Uses caching/persistence'),
        (r'\.watermark\(', 'Implements watermarks'),
        (r'\.window\(', 'Uses windowing operations')
    ]

    passed = []
    failed = []
    for pattern, description in performance_patterns:
        if re.search(pattern, code_content):
            passed.append(description)
        else:
            failed.append(f"Missing: {description}")

    return len(passed) / len(performance_patterns), failed

def evaluate_best_practices(code_content: str) -> Tuple[float, List[str]]:
    best_practices = [
        (r'/\*\*[\s\S]*?\*/', 'Has JavaDoc comments'),
        (r'private\s+static\s+final', 'Uses constants'),
        (r'@Override', 'Uses proper annotations'),
        (r'implements\s+Serializable', 'Implements Serializable'),
        (r'try\s*\{[\s\S]*?}\s*catch', 'Has exception handling'),
        (r'logger\.|LOG\.', 'Uses logging')
    ]

    passed = []
    failed = []
    for pattern, description in best_practices:
        if re.search(pattern, code_content):
            passed.append(description)
        else:
            failed.append(f"Missing: {description}")

    return len(passed) / len(best_practices), failed

def evaluate_error_handling(code_content: str) -> Tuple[float, List[str]]:
    error_patterns = [
        (r'try\s*\{[\s\S]*?}\s*catch', 'Has try-catch blocks'),
        (r'throw\s+new', 'Throws exceptions'),
        (r'finally\s*\{', 'Uses finally blocks'),
        (r'catch\s*\([^)]+\)', 'Catches specific exceptions'),
        (r'logger\.|LOG\.error', 'Logs errors')
    ]

    passed = []
    failed = []
    for pattern, description in error_patterns:
        if re.search(pattern, code_content):
            passed.append(description)
        else:
            failed.append(f"Missing: {description}")

    return len(passed) / len(error_patterns), failed

def evaluate_security(code_content: str) -> Tuple[float, List[str]]:
    security_patterns = [
        (r'SecurityConfig|SecurityContext', 'Has security configuration'),
        (r'encrypt|decrypt', 'Implements encryption'),
        (r'authentication|authorize', 'Has authentication/authorization'),
        (r'sanitize|escape', 'Implements input sanitization'),
        (r'private\s+(?:final\s+)?[A-Za-z]+\s+\w+', 'Uses private fields')
    ]

    passed = []
    failed = []
    for pattern, description in security_patterns:
        if re.search(pattern, code_content):
            passed.append(description)
        else:
            failed.append(f"Missing: {description}")

    return len(passed) / len(security_patterns), failed

def evaluate_scalability(code_content: str) -> Tuple[float, List[str]]:
    scalability_patterns = [
        (r'setParallelism|numPartitions', 'Configurable parallelism'),
        (r'checkpoint|savepoint', 'Has checkpointing'),
        (r'backpressure|buffer', 'Handles backpressure'),
        (r'scale|scaling', 'Has scaling configuration'),
        (r'cluster|distributed', 'Distributed processing support')
    ]

    passed = []
    failed = []
    for pattern, description in scalability_patterns:
        if re.search(pattern, code_content):
            passed.append(description)
        else:
            failed.append(f"Missing: {description}")

    return len(passed) / len(scalability_patterns), failed

def comprehensive_code_evaluation(response_content: str, streaming_system: str) -> CodeEvaluation:
    # Extract code blocks from response
    code_blocks = re.findall(r'```(?:java)?\n(.*?)```', response_content, re.DOTALL)
    code_content = '\n'.join(code_blocks)

    # Evaluate different aspects
    accuracy_score, accuracy_details = evaluate_code_accuracy(code_content, streaming_system)
    performance_score, performance_details = evaluate_performance_patterns(code_content)
    practices_score, practices_details = evaluate_best_practices(code_content)
    error_score, error_details = evaluate_error_handling(code_content)
    security_score, security_details = evaluate_security(code_content)
    scalability_score, scalability_details = evaluate_scalability(code_content)

    # Calculate weighted overall score
    weights = {
        'accuracy': 0.3,
        'performance': 0.2,
        'practices': 0.15,
        'error': 0.15,
        'security': 0.1,
        'scalability': 0.1
    }

    overall_score = sum([
        accuracy_score * weights['accuracy'],
        performance_score * weights['performance'],
        practices_score * weights['practices'],
        error_score * weights['error'],
        security_score * weights['security'],
        scalability_score * weights['scalability']
    ])

    return CodeEvaluation(
        accuracy=accuracy_score * 100,
        performance=performance_score * 100,
        best_practices=practices_score * 100,
        error_handling=error_score * 100,
        security=security_score * 100,
        scalability=scalability_score * 100,
        overall_score=overall_score * 100,
        details={
            'accuracy': accuracy_details,
            'performance': performance_details,
            'best_practices': practices_details,
            'error_handling': error_details,
            'security': security_details,
            'scalability': scalability_details
        }
    )

def print_evaluation_report(eval_result: CodeEvaluation):
    print("\nCode Evaluation Report:")
    print(f"Overall Score: {eval_result.overall_score:.1f}%\n")

    metrics = [
        ("Accuracy", eval_result.accuracy),
        ("Performance", eval_result.performance),
        ("Best Practices", eval_result.best_practices),
        ("Error Handling", eval_result.error_handling),
        ("Security", eval_result.security),
        ("Scalability", eval_result.scalability)
    ]

    for name, score in metrics:
        print(f"{name}: {score:.1f}%")
        if eval_result.details.get(name.lower().replace(' ', '_')):
            for detail in eval_result.details[name.lower().replace(' ', '_')]:
                print(f"  - {detail}")
        print()