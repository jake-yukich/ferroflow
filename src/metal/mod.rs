use metal::{Device, CommandQueue, Library, ComputePipelineState};
use std::sync::Arc;
use metal::MTLSize;
use std::mem::size_of;
use crate::tensor::{Tensor, Shape};
use crate::compute::{ComputeBackend, MetalBackend};

pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
    add_pipeline: ComputePipelineState,
    multiply_pipeline: ComputePipelineState,
    scalar_multiply_pipeline: ComputePipelineState,
    conv2d_pipeline: ComputePipelineState,
}

impl MetalContext {
    pub fn new() -> crate::error::Result<Arc<Self>> {
        let device = Device::system_default()
            .ok_or_else(|| crate::error::FerroFlowError::InitError("No Metal device found".into()))?;
        
        let command_queue = device.new_command_queue();
        
        // Load and compile Metal shader library
        let library = device.new_library_with_source(include_str!("shaders.metal"), &metal::CompileOptions::new())
            .map_err(|e| crate::error::FerroFlowError::MetalError(e.to_string()))?;
        
        // Pipeline states for kernels
        let add_pipeline = Self::create_pipeline(&device, &library, "element_wise_add")?;
        let multiply_pipeline = Self::create_pipeline(&device, &library, "element_wise_multiply")?;
        let scalar_multiply_pipeline = Self::create_pipeline(&device, &library, "scalar_multiply")?;
        let conv2d_pipeline = Self::create_pipeline(&device, &library, "conv2d")?;
        
        Ok(Arc::new(Self {
            device,
            command_queue,
            library,
            add_pipeline,
            multiply_pipeline,
            scalar_multiply_pipeline,
            conv2d_pipeline,
        }))
    }
    
    fn create_pipeline(device: &Device, library: &Library, function_name: &str) -> crate::error::Result<ComputePipelineState> {
        let function = library.get_function(function_name, None)
            .map_err(|e| crate::error::FerroFlowError::MetalError(e.to_string()))?;
            
        device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| crate::error::FerroFlowError::MetalError(e.to_string()))
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    fn get_pipeline_state(&self, name: &str) -> crate::error::Result<ComputePipelineState> {
        match name {
            "conv2d" => {
                let function = self.library.get_function("conv2d", None)
                    .map_err(|e| crate::error::FerroFlowError::MetalError(e.to_string()))?;
                
                self.device.new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| crate::error::FerroFlowError::MetalError(e.to_string()))
            }
            _ => Err(crate::error::FerroFlowError::MetalError("Unknown pipeline state".into()))
        }
    }

    // TODO: Move conv2d implementation to compute/metal.rs
    #[allow(dead_code, unused_variables)]
    fn conv2d(&self, ctx: Arc<MetalContext>, input: &Tensor<MetalBackend>, kernel: &Tensor<MetalBackend>, stride: (usize, usize), padding: (usize, usize)) -> crate::error::Result<Tensor<MetalBackend>> {
        // Implementation moved to compute/metal.rs
        unimplemented!("conv2d should be implemented in compute/metal.rs")
        /*
        let (batch_size, in_channels, in_height, in_width) = input.shape().into_dims4()?;
        let (out_channels, _, kernel_height, kernel_width) = kernel.shape().into_dims4()?;
        
        // Calculate output dimensions
        let out_height = ((in_height + 2 * padding.0 - kernel_height) / stride.0) + 1;
        let out_width = ((in_width + 2 * padding.1 - kernel_width) / stride.1) + 1;
        
        let output_shape = Shape::new(vec![batch_size, out_channels, out_height, out_width]);
        let output = Tensor::<MetalBackend>::zeros(ctx, output_shape)?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // Get the conv2d kernel
        compute_encoder.set_compute_pipeline_state(&self.conv2d_pipeline);

        // Set buffer arguments
        compute_encoder.set_buffer(0, Some(&input.buffer), 0);
        compute_encoder.set_buffer(1, Some(&kernel.buffer), 0);
        compute_encoder.set_buffer(2, Some(&output.buffer), 0);

        // Set uniform values as individual parameters
        let batch_size_u32 = batch_size as u32;
        let in_channels_u32 = in_channels as u32;
        let in_height_u32 = in_height as u32;
        let in_width_u32 = in_width as u32;
        let out_channels_u32 = out_channels as u32;
        let kernel_height_u32 = kernel_height as u32;
        let kernel_width_u32 = kernel_width as u32;
        let out_height_u32 = out_height as u32;
        let out_width_u32 = out_width as u32;
        let stride_h_u32 = stride.0 as u32;
        let stride_w_u32 = stride.1 as u32;
        let pad_h_u32 = padding.0 as u32;
        let pad_w_u32 = padding.1 as u32;
        
        compute_encoder.set_bytes(3, size_of::<u32>() as u64, &batch_size_u32 as *const _ as *const _);
        compute_encoder.set_bytes(4, size_of::<u32>() as u64, &in_channels_u32 as *const _ as *const _);
        compute_encoder.set_bytes(5, size_of::<u32>() as u64, &in_height_u32 as *const _ as *const _);
        compute_encoder.set_bytes(6, size_of::<u32>() as u64, &in_width_u32 as *const _ as *const _);
        compute_encoder.set_bytes(7, size_of::<u32>() as u64, &out_channels_u32 as *const _ as *const _);
        compute_encoder.set_bytes(8, size_of::<u32>() as u64, &kernel_height_u32 as *const _ as *const _);
        compute_encoder.set_bytes(9, size_of::<u32>() as u64, &kernel_width_u32 as *const _ as *const _);
        compute_encoder.set_bytes(10, size_of::<u32>() as u64, &out_height_u32 as *const _ as *const _);
        compute_encoder.set_bytes(11, size_of::<u32>() as u64, &out_width_u32 as *const _ as *const _);
        compute_encoder.set_bytes(12, size_of::<u32>() as u64, &stride_h_u32 as *const _ as *const _);
        compute_encoder.set_bytes(13, size_of::<u32>() as u64, &stride_w_u32 as *const _ as *const _);
        compute_encoder.set_bytes(14, size_of::<u32>() as u64, &pad_h_u32 as *const _ as *const _);
        compute_encoder.set_bytes(15, size_of::<u32>() as u64, &pad_w_u32 as *const _ as *const _);

        // Dispatch the kernel
        let grid_size = MTLSize::new(
            (out_height * out_width) as u64,
            out_channels as u64,
            batch_size as u64
        );
        let thread_group_size = MTLSize::new(64, 1, 1);
        compute_encoder.dispatch_threads(grid_size, thread_group_size);

        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
        */
    }
}

 