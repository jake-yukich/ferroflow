use metal::{Device, CommandQueue, Library, ComputePipelineState};
use std::sync::Arc;

pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
    add_pipeline: ComputePipelineState,
    multiply_pipeline: ComputePipelineState,
    scalar_multiply_pipeline: ComputePipelineState,
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
        
        Ok(Arc::new(Self {
            device,
            command_queue,
            library,
            add_pipeline,
            multiply_pipeline,
            scalar_multiply_pipeline,
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
} 