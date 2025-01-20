#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{CPUBackend, MetalBackend};

    #[test]
    fn test_cpu_operations() -> Result<()> {
        let ctx = CPUBackend::new()?;
        test_backend_operations::<CPUBackend>(ctx)
    }

    #[test]
    fn test_metal_operations() -> Result<()> {
        let ctx = MetalBackend::new()?;
        test_backend_operations::<MetalBackend>(ctx)
    }

    fn test_backend_operations<B: ComputeBackend>(ctx: Arc<B::Context>) -> Result<()> {
        let a = Tensor::new(
            Arc::clone(&ctx),
            Shape::new(vec![2, 2]),
            &[1.0, 2.0, 3.0, 4.0],
        )?;
        
        let b = Tensor::new(
            Arc::clone(&ctx),
            Shape::new(vec![2, 2]),
            &[5.0, 6.0, 7.0, 8.0],
        )?;

        // Test addition
        let c = a.add(&b)?;
        assert_eq!(c.data()?, vec![6.0, 8.0, 10.0, 12.0]);

        // Test multiplication
        let d = a.multiply(&b)?;
        assert_eq!(d.data()?, vec![5.0, 12.0, 21.0, 32.0]);

        // Test scalar multiplication
        let e = a.scalar_multiply(2.0)?;
        assert_eq!(e.data()?, vec![2.0, 4.0, 6.0, 8.0]);

        Ok(())
    }
}