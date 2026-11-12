import os

import numpy as np
import pandas as pd


def test_dataset_exists():
    assert os.path.exists("data/career_guidance_dataset.csv")

def test_data_shape():
    df = pd.read_csv("data/career_guidance_dataset.csv")
    assert df.shape[0] == 20000
    assert df.shape[1] == 39

def test_debate_engine_import():
    from src.debate_agent import MultiPersonaDebateEngine
    engine = MultiPersonaDebateEngine()
    assert engine.entropy_threshold == 1.25
    assert engine.mahalanobis_threshold == 4.0

def test_unit3_pipeline_import():
    from src.unit3_unsupervised import Unit3UnsupervisedPipeline
    u3 = Unit3UnsupervisedPipeline()
    assert u3 is not None

def test_unit4_sequential_import():
    from src.unit4_sequential import CareerMarkovChain
    mc = CareerMarkovChain()
    assert len(mc.states) == 5
    stat = mc.compute_stationary_distribution()
    assert len(stat) == 5
    assert np.isclose(np.sum(stat), 1.0)

def test_polar_radar_dimensions():
    categories = ["Operating Systems", "Algorithms", "Programming", "Computer Networks", "Mathematics", "Coding Skills"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values = [90, 85, 95, 80, 88, 90]
    values += values[:1]
    assert len(angles) == len(values) == 7

def test_debate_scenarios_logic():
    from src.debate_agent import MultiPersonaDebateEngine
    engine = MultiPersonaDebateEngine(entropy_threshold=1.20, mahalanobis_threshold=3.5)
    
    # Test Scenario B: Mahalanobis outlier
    res_b = engine.execute_debate(
        student_id="test_b",
        student_profile={},
        unit3_metrics={
            "primary_cluster": 1,
            "soft_responsibilities": [0.1, 0.8, 0.1],
            "assignment_entropy": 0.6,
            "mahalanobis_distance": 4.5
        },
        unit4_metrics={"viterbi_path": ["Novice", "Competent", "Senior"]}
    )
    assert not res_b["persona_critic"]["is_approved"]
    assert "Conditional" in res_b["consensus_synthesis"]["recommendation"]

    # Test Scenario C: Shannon entropy override
    res_c = engine.execute_debate(
        student_id="test_c",
        student_profile={},
        unit3_metrics={
            "primary_cluster": 0,
            "soft_responsibilities": [0.34, 0.33, 0.33],
            "assignment_entropy": 1.35,
            "mahalanobis_distance": 2.1
        },
        unit4_metrics={"viterbi_path": ["Novice", "Competent", "Competent"]}
    )
    assert res_c["persona_ethics"]["override_triggered"]
    assert "Hybrid" in res_c["consensus_synthesis"]["recommendation"]

