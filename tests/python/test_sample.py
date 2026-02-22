"""Python integration tests for ucc.compile and ucc.sample."""

import ucc


class TestCompile:
    """Tests for ucc.compile()."""

    def test_compile_simple(self) -> None:
        """Compile a simple circuit."""
        prog = ucc.compile("H 0\nT 0\nM 0")
        assert prog.peak_rank == 1
        assert prog.num_measurements == 1
        assert prog.num_instructions >= 1

    def test_compile_pure_clifford(self) -> None:
        """Pure Clifford circuit has peak_rank 0."""
        prog = ucc.compile("H 0\nCX 0 1\nM 0\nM 1")
        assert prog.peak_rank == 0
        assert prog.num_measurements == 2

    def test_compile_multiple_t_gates(self) -> None:
        """Multiple independent T gates increase peak_rank."""
        prog = ucc.compile("""
            H 0
            H 1
            T 0
            T 1
        """)
        assert prog.peak_rank == 2


class TestSample:
    """Tests for ucc.sample()."""

    def test_sample_deterministic_zero(self) -> None:
        """Measurement of |0⟩ always gives 0."""
        prog = ucc.compile("M 0")
        results = ucc.sample(prog, 100, seed=42)
        for shot in results:
            assert shot[0] == 0

    def test_sample_deterministic_one(self) -> None:
        """Measurement of |1⟩ always gives 1."""
        prog = ucc.compile("X 0\nM 0")
        results = ucc.sample(prog, 100, seed=42)
        for shot in results:
            assert shot[0] == 1

    def test_sample_superposition(self) -> None:
        """|+⟩ state gives roughly 50/50 distribution."""
        prog = ucc.compile("H 0\nM 0")
        results = ucc.sample(prog, 1000, seed=42)
        zeros = sum(1 for r in results if r[0] == 0)
        ones = sum(1 for r in results if r[0] == 1)
        # Allow 10% tolerance
        assert 400 < zeros < 600
        assert 400 < ones < 600

    def test_sample_bell_state_correlated(self) -> None:
        """Bell state measurements are always correlated."""
        prog = ucc.compile("""
            H 0
            CX 0 1
            M 0
            M 1
        """)
        results = ucc.sample(prog, 500, seed=99)
        for shot in results:
            assert shot[0] == shot[1], f"Bell state not correlated: {shot}"

    def test_sample_reproducible(self) -> None:
        """Same seed produces same results."""
        prog = ucc.compile("H 0\nM 0")
        results1 = ucc.sample(prog, 100, seed=12345)
        results2 = ucc.sample(prog, 100, seed=12345)
        assert results1 == results2

    def test_sample_different_seeds(self) -> None:
        """Different seeds produce different results."""
        prog = ucc.compile("H 0\nM 0")
        results1 = ucc.sample(prog, 100, seed=1)
        results2 = ucc.sample(prog, 100, seed=2)
        # With 100 random bits, probability of match is 2^-100
        assert results1 != results2

    def test_sample_shape(self) -> None:
        """Results have correct shape."""
        prog = ucc.compile("H 0\nM 0\nH 1\nM 1")
        results = ucc.sample(prog, 50, seed=0)
        assert len(results) == 50
        for shot in results:
            assert len(shot) == 2

    def test_sample_reset_works(self) -> None:
        """Reset correctly resets to |0⟩."""
        prog = ucc.compile("""
            X 0
            R 0
            M 0
        """)
        results = ucc.sample(prog, 100, seed=42)
        # Second measurement (after reset) should always be 0
        for shot in results:
            assert shot[1] == 0, f"Reset failed: {shot}"
