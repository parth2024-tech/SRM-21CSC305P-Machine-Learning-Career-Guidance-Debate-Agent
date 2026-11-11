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
