use super::ComputeBackend;
use crate::error::{Result, FerroFlowError};
use metal::{self, Device, CommandQueue, Library, ComputePipelineState, Buffer};
use std::sync::Arc;

const TILE_SIZE: u32 = 16;

#[derive(Debug)]
pub struct MetalContext {
    pub(crate) device: Device,
    pub(crate) command_queue: CommandQueue,
    pub(crate) library: Library,
    pub(crate) add_pipeline: ComputePipelineState,
    pub(crate) multiply_pipeline: ComputePipelineState,
    pub(crate) scalar_multiply_pipeline: ComputePipelineState,
    pub(crate) matmul_pipeline: ComputePipelineState,
    pub(crate) matmul_tiled_pipeline: ComputePipelineState,
    pub(crate) matmul_batched_pipeline: ComputePipelineState,
    pub(crate) matmul_batched_tiled_pipeline: ComputePipelineState,
    pub(crate) matmul_transposed_pipeline: ComputePipelineState,
    pub(crate) matmul_transposed_tiled_pipeline: ComputePipelineState,
}

pub struct MetalBackend;

impl MetalContext {
    fn create_pipeline(device: &Device, library: &Library, function_name: &str) -> Result<ComputePipelineState> {
        let function = library.get_function(function_name, None)
            .map_err(|e| FerroFlowError::MetalError(e.to_string()))?;
            
        device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| FerroFlowError::MetalError(e.to_string()))
    }

    pub(crate) fn create_matmul_pipelines(device: &Device, library: &Library) 
        -> Result<(ComputePipelineState, ComputePipelineState, ComputePipelineState, ComputePipelineState, 
                  ComputePipelineState, ComputePipelineState)> 
    {
        let basic = Self::create_pipeline(device, library, "matmul")?;
        let tiled = Self::create_pipeline(device, library, "matmul_tiled")?;
        let batched = Self::create_pipeline(device, library, "matmul_batched")?;
        let batched_tiled = Self::create_pipeline(device, library, "matmul_batched_tiled")?;
        let transposed = Self::create_pipeline(device, library, "matmul_transposed")?;
        let transposed_tiled = Self::create_pipeline(device, library, "matmul_transposed_tiled")?;
        Ok((basic, tiled, batched, batched_tiled, transposed, transposed_tiled))
    }
}

impl ComputeBackend for MetalBackend {
    type Buffer = Buffer;
    type Context = MetalContext;

    fn new() -> Result<Arc<Self::Context>> {
        let device = Device::system_default()
            .ok_or_else(|| FerroFlowError::InitError("No Metal device found".into()))?;
        
        let command_queue = device.new_command_queue();
        
        let library = device.new_library_with_source(include_str!("../metal/shaders.metal"), &metal::CompileOptions::new())
            .map_err(|e| FerroFlowError::MetalError(e.to_string()))?;
        
        let add_pipeline = MetalContext::create_pipeline(&device, &library, "element_wise_add")?;
        let multiply_pipeline = MetalContext::create_pipeline(&device, &library, "element_wise_multiply")?;
        let scalar_multiply_pipeline = MetalContext::create_pipeline(&device, &library, "scalar_multiply")?;
        let (matmul_pipeline, matmul_tiled_pipeline, matmul_batched_pipeline, matmul_batched_tiled_pipeline, matmul_transposed_pipeline, matmul_transposed_tiled_pipeline) = MetalContext::create_matmul_pipelines(&device, &library)?;
        
        Ok(Arc::new(MetalContext {
            device,
            command_queue,
            library,
            add_pipeline,
            multiply_pipeline,
            scalar_multiply_pipeline,
            matmul_pipeline,
            matmul_tiled_pipeline,
            matmul_batched_pipeline,
            matmul_batched_tiled_pipeline,
            matmul_transposed_pipeline,
            matmul_transposed_tiled_pipeline,
        }))
    }

    fn allocate_buffer(ctx: &Self::Context, size: usize, data: Option<&[f32]>) -> Result<Self::Buffer> {
        let buffer_size = (size * std::mem::size_of::<f32>()) as u64;
        
        match data {
            Some(data) => {
                if data.len() != size {
                    return Err(FerroFlowError::BufferError("Data size mismatch".into()));
                }
                Ok(ctx.device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    buffer_size,
                    metal::MTLResourceOptions::StorageModeShared,
                ))
            },
            None => Ok(ctx.device.new_buffer(
                buffer_size,
                metal::MTLResourceOptions::StorageModeShared,
            ))
        }
    }

    fn read_buffer(ctx: &Self::Context, buffer: &Self::Buffer) -> Result<Vec<f32>> {
        let contents = buffer.contents() as *const f32;
        let size = buffer.length() as usize / std::mem::size_of::<f32>();
        let mut result = Vec::with_capacity(size);
        
        unsafe {
            std::ptr::copy_nonoverlapping(contents, result.as_mut_ptr(), size);
            result.set_len(size);
        }
        
        Ok(result)
    }

    fn element_wise_add(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        size: usize,
    ) -> Result<Self::Buffer> {
        let result_buffer = Self::allocate_buffer(ctx, size, None)?;
        
        let command_buffer = ctx.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        compute_encoder.set_compute_pipeline_state(&ctx.add_pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);
        
        let grid_size = metal::MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        
        compute_encoder.dispatch_threads(grid_size, threadgroup_size);
        compute_encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result_buffer)
    }

    fn element_wise_multiply(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        size: usize,
    ) -> Result<Self::Buffer> {
        let result_buffer = Self::allocate_buffer(ctx, size, None)?;
        
        let command_buffer = ctx.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        compute_encoder.set_compute_pipeline_state(&ctx.multiply_pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);
        
        let grid_size = metal::MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        
        compute_encoder.dispatch_threads(grid_size, threadgroup_size);
        compute_encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result_buffer)
    }

    fn scalar_multiply(
        ctx: &Self::Context,
        input: &Self::Buffer,
        scalar: f32,
        size: usize,
    ) -> Result<Self::Buffer> {
        let result_buffer = Self::allocate_buffer(ctx, size, None)?;
        
        let command_buffer = ctx.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        compute_encoder.set_compute_pipeline_state(&ctx.scalar_multiply_pipeline);
        compute_encoder.set_buffer(0, Some(input), 0);
        compute_encoder.set_buffer(1, Some(&result_buffer), 0);
        compute_encoder.set_bytes(2, std::mem::size_of::<f32>() as u64, &scalar as *const f32 as *const _);
        
        let grid_size = metal::MTLSize::new(size as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        
        compute_encoder.dispatch_threads(grid_size, threadgroup_size);
        compute_encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result_buffer)
    }

    fn matmul(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        m: usize,
        n: usize,
        k: usize
    ) -> Result<Self::Buffer> {
        let result_buffer = Self::allocate_buffer(ctx, m * n, None)?;
        
        let command_buffer = ctx.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        let use_tiled = m >= TILE_SIZE as usize && n >= TILE_SIZE as usize && k >= TILE_SIZE as usize;
        let pipeline = if use_tiled {
            &ctx.matmul_tiled_pipeline
        } else {
            &ctx.matmul_pipeline
        };
        
        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);
        
        compute_encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(m as u32) as *const u32 as *const _);
        compute_encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
        compute_encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(k as u32) as *const u32 as *const _);
        
        let grid_size = metal::MTLSize::new(n as u64, m as u64, 1);
        let threadgroup_size = metal::MTLSize::new(
            TILE_SIZE as u64,
            TILE_SIZE as u64,
            1
        );
        
        compute_encoder.dispatch_threads(grid_size, threadgroup_size);
        compute_encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result_buffer)
    }

    fn matmul_batched(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize
    ) -> Result<Self::Buffer> {
        let result_buffer = Self::allocate_buffer(ctx, batch_size * m * n, None)?;
        
        let command_buffer = ctx.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        let use_tiled = m >= TILE_SIZE as usize && n >= TILE_SIZE as usize && k >= TILE_SIZE as usize;
        let pipeline = if use_tiled {
            &ctx.matmul_batched_tiled_pipeline
        } else {
            &ctx.matmul_batched_pipeline
        };
        
        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);
        
        compute_encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(m as u32) as *const u32 as *const _);
        compute_encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
        compute_encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(k as u32) as *const u32 as *const _);
        compute_encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &(batch_size as u32) as *const u32 as *const _);
        
        let grid_size = metal::MTLSize::new(n as u64, m as u64, batch_size as u64);
        let threadgroup_size = metal::MTLSize::new(
            TILE_SIZE as u64,
            TILE_SIZE as u64,
            1
        );
        
        compute_encoder.dispatch_threads(grid_size, threadgroup_size);
        compute_encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result_buffer)
    }

    fn matmul_transposed(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool
    ) -> Result<Self::Buffer> {
        let command_buffer = ctx.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // Choose between tiled and non-tiled based on matrix size
        let pipeline = if m >= TILE_SIZE && n >= TILE_SIZE && k >= TILE_SIZE {
            &ctx.matmul_transposed_tiled_pipeline
        } else {
            &ctx.matmul_transposed_pipeline
        };

        let result_buffer = ctx.device.new_buffer(
            (m * n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared
        );

        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(0, Some(a), 0);
        compute_encoder.set_buffer(1, Some(b), 0);
        compute_encoder.set_buffer(2, Some(&result_buffer), 0);
        
        compute_encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(m as u32) as *const u32 as *const _);
        compute_encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(n as u32) as *const u32 as *const _);
        compute_encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &(k as u32) as *const u32 as *const _);
        compute_encoder.set_bytes(6, std::mem::size_of::<bool>() as u64, &transpose_a as *const bool as *const _);
        compute_encoder.set_bytes(7, std::mem::size_of::<bool>() as u64, &transpose_b as *const bool as *const _);

        let grid_size = metal::MTLSize::new(n as u64, m as u64, 1);
        let threadgroup_size = metal::MTLSize::new(
            TILE_SIZE as u64,
            TILE_SIZE as u64,
            1
        );

        compute_encoder.dispatch_threads(grid_size, threadgroup_size);
        compute_encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result_buffer)
    }

    fn synchronize(ctx: &Self::Context) -> Result<()> {
        Ok(())
    }
} 