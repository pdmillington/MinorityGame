#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:12:24 2026

@author: petermillington
"""

"""
Unit tests for PopulationFactory
"""
import pytest
from core.population_factory import PopulationFactory, Cohort


class TestPopulationFactorySimpleCohorts:
    """Test population creation from simple cohort specifications"""
    
    def test_single_cohort(self):
        """Test creating population with single cohort"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 100, "memory": 3, "payoff": "BinaryMG", "strategies": 2}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 100
        assert len(meta["cohorts"]) == 1
        assert meta["cohorts"][0].count == 100
        assert meta["cohorts"][0].memory == 3
        assert meta["cohorts"][0].payoff == "BinaryMG"
        assert meta["cohorts"][0].strategies == 2
    
    def test_multiple_cohorts(self):
        """Test creating population with multiple cohorts"""
        spec = {
            "total": 150,
            "cohorts": [
                {"count": 50, "memory": 3, "payoff": "BinaryMG", "strategies": 2},
                {"count": 50, "memory": 5, "payoff": "ScaledMG", "strategies": 4},
                {"count": 50, "memory": 7, "payoff": "DollarGame", "strategies": 6}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 150
        assert len(meta["cohorts"]) == 3
        assert sum(c.count for c in meta["cohorts"]) == 150


class TestPopulationFactoryNoiseCohorts:
    """Test population creation with noise traders"""
    
    def test_noise_only(self):
        """Test population with only noise traders"""
        spec = {
            "total": 50,
            "cohorts": [
                {"count": 50, "agent_type": "noise"}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 50
        assert meta["cohorts"][0].agent_type == "noise"
        assert meta["cohorts"][0].memory == 1  # Noise traders default to memory=1
    
    def test_mixed_strategic_and_noise(self):
        """Test population with both strategic and noise traders"""
        spec = {
            "total": 110,
            "cohorts": [
                {"count": 100, "memory": 5, "payoff": "BinaryMG", "strategies": 2, "agent_type": "strategic"},
                {"count": 10, "agent_type": "noise"}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 110
        assert len(meta["cohorts"]) == 2
        assert meta["cohorts"][0].agent_type == "strategic"
        assert meta["cohorts"][1].agent_type == "noise"
    
    def test_default_agent_type(self):
        """Test that agent_type defaults to 'strategic' if not specified"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 100, "memory": 3, "payoff": "BinaryMG", "strategies": 2}
                # No agent_type specified
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["cohorts"][0].agent_type == "strategic"


class TestPopulationFactoryProportionalCounts:
    """Test population creation with proportional counts"""
    
    def test_simple_proportions(self):
        """Test proportional cohort counts"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 0.6, "memory": 3, "payoff": "BinaryMG", "strategies": 2},
                {"count": 0.4, "memory": 5, "payoff": "ScaledMG", "strategies": 4}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 100
        counts = [c.count for c in meta["cohorts"]]
        assert 60 in counts, "Should have 60% of 100 = 60"
        assert 40 in counts, "Should have 40% of 100 = 40"
    
    def test_proportions_with_remainder(self):
        """Test that proportional counts handle remainders correctly"""
        spec = {
            "total": 101,
            "cohorts": [
                {"count": 0.333, "memory": 3, "payoff": "BinaryMG", "strategies": 2},
                {"count": 0.333, "memory": 5, "payoff": "ScaledMG", "strategies": 4},
                {"count": 0.334, "memory": 7, "payoff": "DollarGame", "strategies": 6}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 101
        total_count = sum(c.count for c in meta["cohorts"])
        assert total_count == 101, "Total should match exactly"
    
    def test_mixed_absolute_and_proportional(self):
        """Test mixing absolute and proportional counts"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 50, "memory": 3, "payoff": "BinaryMG", "strategies": 2},  # Absolute
                {"count": 0.5, "memory": 5, "payoff": "ScaledMG", "strategies": 4}  # Proportional
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 100
        counts = [c.count for c in meta["cohorts"]]
        assert 50 in counts
        # Remaining 50 should go to proportional cohort


class TestPopulationFactoryPositionLimits:
    """Test position limit handling"""
    
    def test_position_limits_specified(self):
        """Test that position limits are correctly set"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 50, "memory": 3, "payoff": "BinaryMG", "strategies": 2, "position_limit": 10},
                {"count": 50, "memory": 5, "payoff": "ScaledMG", "strategies": 4, "position_limit": 20}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["cohorts"][0].position_limit == 10
        assert meta["cohorts"][1].position_limit == 20
    
    def test_position_limit_zero(self):
        """Test that position_limit=0 is preserved (means no limit)"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 100, "memory": 3, "payoff": "BinaryMG", "strategies": 2, "position_limit": 0}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["cohorts"][0].position_limit == 0
    
    def test_position_limit_default(self):
        """Test that position_limit defaults to 0 if not specified"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 100, "memory": 3, "payoff": "BinaryMG", "strategies": 2}
                # No position_limit specified
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        # Default should be 0 (no limit) or None
        assert meta["cohorts"][0].position_limit in [0, None]


class TestPopulationFactoryValidation:
    """Test validation and error handling"""
    
    def test_missing_memory_error(self):
        """Test that missing memory for strategic agent raises error"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 100, "payoff": "BinaryMG", "strategies": 2, "agent_type": "strategic"}
                # Missing memory!
            ]
        }
        
        factory = PopulationFactory(spec)
        with pytest.raises(ValueError, match="missing memory"):
            factory.build()
    
    def test_proportional_without_total_error(self):
        """Test that proportional counts without total raises error"""
        spec = {
            # No total specified!
            "cohorts": [
                {"count": 0.5, "memory": 3, "payoff": "BinaryMG", "strategies": 2},
                {"count": 0.5, "memory": 5, "payoff": "ScaledMG", "strategies": 4}
            ]
        }
        
        factory = PopulationFactory(spec)
        with pytest.raises(ValueError, match="Proportional counts require 'total'"):
            factory.build()
    
    def test_absolute_exceeds_total_error(self):
        """Test that absolute counts exceeding total raises error"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 60, "memory": 3, "payoff": "BinaryMG", "strategies": 2},
                {"count": 50, "memory": 5, "payoff": "ScaledMG", "strategies": 4}
                # Total = 110, exceeds specified 100!
            ]
        }
        
        factory = PopulationFactory(spec)
        with pytest.raises(ValueError, match="exceed total"):
            factory.build()
    
    def test_zero_count_cohorts_ignored(self):
        """Test that cohorts with zero count are ignored"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 100, "memory": 3, "payoff": "BinaryMG", "strategies": 2},
                {"count": 0, "memory": 5, "payoff": "ScaledMG", "strategies": 4}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        # Should only have 1 cohort (zero count ignored)
        assert len(meta["cohorts"]) == 1
        assert meta["cohorts"][0].count == 100


class TestPopulationFactoryMetadata:
    """Test metadata returned by factory"""
    
    def test_metadata_structure(self):
        """Test that metadata has expected structure"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 100, "memory": 3, "payoff": "BinaryMG", "strategies": 2}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        # Check all expected keys are present
        assert "N" in meta
        assert "memory" in meta
        assert "strategies" in meta
        assert "payoff_code" in meta
        assert "cohort_id" in meta
        assert "position_limit" in meta
        assert "cohorts" in meta
        assert "payoff_map" in meta
    
    def test_payoff_map(self):
        """Test that payoff_map is correctly populated"""
        spec = {
            "total": 100,
            "cohorts": [
                {"count": 100, "memory": 3, "payoff": "BinaryMG", "strategies": 2}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert "payoff_map" in meta
        assert "BinaryMG" in meta["payoff_map"]
        assert "ScaledMG" in meta["payoff_map"]
        assert "DollarGame" in meta["payoff_map"]
    
    def test_arrays_have_correct_size(self):
        """Test that numpy arrays have correct size"""
        spec = {
            "total": 150,
            "cohorts": [
                {"count": 100, "memory": 3, "payoff": "BinaryMG", "strategies": 2},
                {"count": 50, "memory": 5, "payoff": "ScaledMG", "strategies": 4}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert len(meta["memory"]) == 150
        assert len(meta["strategies"]) == 150
        assert len(meta["payoff_code"]) == 150
        assert len(meta["cohort_id"]) == 150
        assert len(meta["position_limit"]) == 150


class TestPopulationFactoryCohortDataclass:
    """Test Cohort dataclass functionality"""
    
    def test_cohort_creation(self):
        """Test creating Cohort instances"""
        cohort = Cohort(
            count=100,
            memory=5,
            payoff="BinaryMG",
            strategies=2,
            position_limit=10,
            agent_type="strategic"
        )
        
        assert cohort.count == 100
        assert cohort.memory == 5
        assert cohort.payoff == "BinaryMG"
        assert cohort.strategies == 2
        assert cohort.position_limit == 10
        assert cohort.agent_type == "strategic"
    
    def test_cohort_defaults(self):
        """Test Cohort default values"""
        cohort = Cohort(count=100)
        
        assert cohort.count == 100
        assert cohort.memory is None
        assert cohort.payoff is None
        assert cohort.strategies is None
        assert cohort.position_limit == 0
        assert cohort.agent_type == "strategic"
    
    def test_cohort_frozen(self):
        """Test that Cohort is frozen (immutable)"""
        cohort = Cohort(count=100, memory=5)
        
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            cohort.count = 200


class TestPopulationFactoryEdgeCases:
    """Test edge cases"""
    
    def test_single_agent(self):
        """Test population with single agent"""
        spec = {
            "total": 1,
            "cohorts": [
                {"count": 1, "memory": 3, "payoff": "BinaryMG", "strategies": 2}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 1
    
    def test_large_population(self):
        """Test large population"""
        spec = {
            "total": 10000,
            "cohorts": [
                {"count": 10000, "memory": 5, "payoff": "BinaryMG", "strategies": 2}
            ]
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 10000
        assert len(meta["memory"]) == 10000
    
    def test_many_small_cohorts(self):
        """Test population with many small cohorts"""
        cohorts = [
            {"count": 1, "memory": i % 5 + 2, "payoff": "BinaryMG", "strategies": 2}
            for i in range(100)
        ]
        
        spec = {
            "total": 100,
            "cohorts": cohorts
        }
        
        factory = PopulationFactory(spec)
        meta = factory.build()
        
        assert meta["N"] == 100
        assert len(meta["cohorts"]) == 100