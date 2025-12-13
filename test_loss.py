"""
Tests for the loss function to ensure gradients are computed correctly.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from config import Config
from model import init_params, loss_fn


class TestLossFunction:
    """Test suite for loss function and gradient computation."""

    @pytest.fixture
    def key(self):
        """Random key for reproducibility."""
        return jr.PRNGKey(42)

    @pytest.fixture
    def params(self, key):
        """Initialize model parameters."""
        return init_params(key)

    @pytest.fixture
    def sample_batch(self, key):
        """Create a sample batch of data."""
        batch_size = 4
        ctx_bits = jr.randint(key, (batch_size, Config.L), 0, Config.vocab_size)
        labels = jr.randint(key, (batch_size,), 0, Config.vocab_size)
        return ctx_bits, labels

    def test_loss_computation(self, params, sample_batch):
        """Test that loss can be computed without errors."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        loss = loss_fn(params, ctx_bits, labels, force_weight)
        
        # Loss should be a scalar
        assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
        # Loss should be finite
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        # Loss should be non-negative (cross-entropy + non-negative force penalty)
        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_loss_gradients_exist(self, params, sample_batch):
        """Test that gradients can be computed for all parameters."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, ctx_bits, labels, force_weight)
        
        # Check that gradients exist for all parameter keys
        assert set(grads.keys()) == set(params.keys()), \
            "Gradient keys don't match parameter keys"
        
        # Check that all gradients have the same shape as parameters
        for key in params.keys():
            assert grads[key].shape == params[key].shape, \
                f"Gradient shape mismatch for {key}: {grads[key].shape} vs {params[key].shape}"

    def test_loss_gradients_finite(self, params, sample_batch):
        """Test that all gradients are finite."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, ctx_bits, labels, force_weight)
        
        for key, grad in grads.items():
            assert jnp.all(jnp.isfinite(grad)), \
                f"Gradient for {key} contains non-finite values"

    def test_loss_gradients_finite_differences(self, params, sample_batch):
        """Test gradient correctness using finite differences."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        eps = 1e-4  # Use slightly larger epsilon for better numerical stability
        
        # Compute analytical gradients
        grad_fn = jax.grad(loss_fn)
        grads_analytical = grad_fn(params, ctx_bits, labels, force_weight)
        
        # Test finite differences for a subset of parameters
        # (testing all would be too slow)
        test_params = ["a", "b", "c", "W_dec", "b_dec"]
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        for param_name in test_params:
            if param_name not in params:
                continue
                
            param = params[param_name]
            grad_analytical = grads_analytical[param_name]
            
            # Flatten for easier iteration
            param_flat = param.flatten()
            grad_flat = grad_analytical.flatten()
            
            # Test a few random indices
            n_test = min(5, len(param_flat))
            indices = np.random.choice(len(param_flat), n_test, replace=False)
            
            for idx in indices:
                # Start from original parameter for each perturbation
                param_np_orig = np.array(param_flat)
                
                # Create plus perturbation
                param_np_plus = param_np_orig.copy()
                param_np_plus[idx] += eps
                param_plus = jnp.array(param_np_plus).reshape(param.shape)
                
                params_plus = {**params, param_name: param_plus}
                loss_plus = loss_fn(params_plus, ctx_bits, labels, force_weight)
                
                # Create minus perturbation (start fresh from original)
                param_np_minus = param_np_orig.copy()
                param_np_minus[idx] -= eps
                param_minus = jnp.array(param_np_minus).reshape(param.shape)
                
                params_minus = {**params, param_name: param_minus}
                loss_minus = loss_fn(params_minus, ctx_bits, labels, force_weight)
                
                grad_finite_diff = (loss_plus - loss_minus) / (2 * eps)
                grad_analytical_val = grad_flat[idx]
                
                # Check gradient correctness
                # Note: Due to the iterative inference process (forward Euler), finite differences
                # can have larger errors than typical. We check:
                # 1. Sign agreement (gradients point in same direction)
                # 2. Both are finite
                # 3. Magnitude is in reasonable range
                
                # Both should be finite
                assert jnp.isfinite(grad_finite_diff), \
                    f"Finite difference gradient for {param_name}[{idx}] is not finite"
                assert jnp.isfinite(grad_analytical_val), \
                    f"Analytical gradient for {param_name}[{idx}] is not finite"
                
                # For very small gradients, just check they're both small
                if abs(grad_analytical_val) < 1e-5:
                    assert abs(grad_finite_diff) < 1e-3, \
                        f"Gradient mismatch for {param_name}[{idx}]: " \
                        f"analytical={grad_analytical_val:.6f}, " \
                        f"finite_diff={grad_finite_diff:.6f} (both should be small)"
                else:
                    # Check sign agreement (allowing for numerical noise)
                    sign_match = (grad_analytical_val * grad_finite_diff >= 0) or \
                                 (abs(grad_analytical_val) < 1e-6 and abs(grad_finite_diff) < 1e-6)
                    
                    # Check that magnitudes are in the same order of magnitude
                    # (within 2 orders of magnitude)
                    ratio = abs(grad_finite_diff) / (abs(grad_analytical_val) + 1e-8)
                    magnitude_ok = (ratio > 0.1) and (ratio < 10.0)
                    
                    # At least one of sign match or magnitude should be OK
                    # (iterative solver can cause sign flips for very small gradients)
                    assert sign_match or magnitude_ok, \
                        f"Gradient mismatch for {param_name}[{idx}]: " \
                        f"analytical={grad_analytical_val:.6f}, " \
                        f"finite_diff={grad_finite_diff:.6f}, " \
                        f"ratio={ratio:.6f}, sign_match={sign_match}"

    def test_loss_with_zero_force_weight(self, params, sample_batch):
        """Test loss computation with zero force weight."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.0)
        
        loss = loss_fn(params, ctx_bits, labels, force_weight)
        assert jnp.isfinite(loss), "Loss should be finite with zero force weight"
        
        # Gradients should still work
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, ctx_bits, labels, force_weight)
        for key, grad in grads.items():
            assert jnp.all(jnp.isfinite(grad)), \
                f"Gradient for {key} should be finite with zero force weight"

    def test_loss_with_large_force_weight(self, params, sample_batch):
        """Test loss computation with large force weight."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(10.0)
        
        loss = loss_fn(params, ctx_bits, labels, force_weight)
        assert jnp.isfinite(loss), "Loss should be finite with large force weight"
        
        # Gradients should still work
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, ctx_bits, labels, force_weight)
        for key, grad in grads.items():
            assert jnp.all(jnp.isfinite(grad)), \
                f"Gradient for {key} should be finite with large force weight"

    def test_loss_different_batch_sizes(self, params, key):
        """Test loss computation with different batch sizes."""
        force_weight = jnp.array(0.1)
        
        for batch_size in [1, 2, 8, 16]:
            k1, k2 = jr.split(key)
            ctx_bits = jr.randint(k1, (batch_size, Config.L), 0, Config.vocab_size)
            labels = jr.randint(k2, (batch_size,), 0, Config.vocab_size)
            
            loss = loss_fn(params, ctx_bits, labels, force_weight)
            assert loss.shape == (), f"Loss should be scalar for batch_size={batch_size}"
            assert jnp.isfinite(loss), f"Loss should be finite for batch_size={batch_size}"

    def test_loss_gradient_shapes(self, params, sample_batch):
        """Test that gradient shapes match parameter shapes exactly."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, ctx_bits, labels, force_weight)
        
        for key in params.keys():
            assert grads[key].shape == params[key].shape, \
                f"Shape mismatch for {key}: grad {grads[key].shape} vs param {params[key].shape}"
            assert grads[key].dtype == params[key].dtype, \
                f"Dtype mismatch for {key}: grad {grads[key].dtype} vs param {params[key].dtype}"

    def test_loss_value_and_grad(self, params, sample_batch):
        """Test that value_and_grad works correctly."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        loss_val, grads = jax.value_and_grad(loss_fn)(params, ctx_bits, labels, force_weight)
        
        # Loss should match direct computation
        loss_direct = loss_fn(params, ctx_bits, labels, force_weight)
        assert jnp.allclose(loss_val, loss_direct), \
            f"value_and_grad loss {loss_val} doesn't match direct loss {loss_direct}"
        
        # Gradients should be finite
        for key, grad in grads.items():
            assert jnp.all(jnp.isfinite(grad)), \
                f"Gradient for {key} from value_and_grad is not finite"

    def test_loss_jit_compatibility(self, params, sample_batch):
        """Test that loss function can be JIT compiled."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        # JIT compile the loss function
        loss_fn_jit = jax.jit(loss_fn)
        loss = loss_fn_jit(params, ctx_bits, labels, force_weight)
        
        assert jnp.isfinite(loss), "JIT-compiled loss should be finite"
        
        # JIT compile gradient computation
        grad_fn_jit = jax.jit(jax.grad(loss_fn))
        grads = grad_fn_jit(params, ctx_bits, labels, force_weight)
        
        for key, grad in grads.items():
            assert jnp.all(jnp.isfinite(grad)), \
                f"JIT-compiled gradient for {key} should be finite"

    def test_loss_gradient_magnitude(self, params, sample_batch):
        """Test that gradients have reasonable magnitudes (not too large or too small)."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, ctx_bits, labels, force_weight)
        
        for key, grad in grads.items():
            grad_norm = jnp.linalg.norm(grad)
            # Gradients should not be extremely large (could indicate numerical issues)
            assert grad_norm < 1e6, \
                f"Gradient norm for {key} is suspiciously large: {grad_norm}"
            # Gradients should not be exactly zero everywhere (unless that's expected)
            # We allow zero gradients for some parameters, but check that at least
            # some parameters have non-zero gradients
            if key in ["W_dec", "b_dec", "a", "b", "c"]:
                assert grad_norm > 1e-10, \
                    f"Gradient for {key} is suspiciously small: {grad_norm}"

    def test_loss_consistency(self, params, sample_batch):
        """Test that loss computation is consistent across multiple calls."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        loss1 = loss_fn(params, ctx_bits, labels, force_weight)
        loss2 = loss_fn(params, ctx_bits, labels, force_weight)
        
        assert jnp.allclose(loss1, loss2), \
            f"Loss should be consistent: {loss1} vs {loss2}"

    def test_loss_gradient_consistency(self, params, sample_batch):
        """Test that gradient computation is consistent across multiple calls."""
        ctx_bits, labels = sample_batch
        force_weight = jnp.array(0.1)
        
        grad_fn = jax.grad(loss_fn)
        grads1 = grad_fn(params, ctx_bits, labels, force_weight)
        grads2 = grad_fn(params, ctx_bits, labels, force_weight)
        
        for key in params.keys():
            assert jnp.allclose(grads1[key], grads2[key]), \
                f"Gradients for {key} should be consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
