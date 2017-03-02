// Header for this file
#include "openclbase.h"
// C System-Headers
//
// C++ System headers
#include <string>
#include <iostream>
// Boost Headers
#include <boost/lexical_cast.hpp>  //lexical cast (unsurprisingly)
// Miscellaneous Headers
//
//Headers for this project
#include "../Generic/debug.h"

namespace ocl {

std::vector<cl::Platform> OpenCLBase::all_platforms;
cl::Platform OpenCLBase::default_platform;
std::vector<cl::Device> OpenCLBase::all_devices;
cl::Device OpenCLBase::current_device;
cl::Context OpenCLBase::context;
cl::CommandQueue OpenCLBase::command_queue;
bool OpenCLBase::initalized = false;

void OpenCLBase::SetUp( uint platform_number, uint device_number ) {

    if( DEBUG_ON ) {
        //Force kernels to be compiled each time
        setenv("CUDA_CACHE_DISABLE", "1", 1);
        std::cout << "OpenCL Kernel caching disabled." << std::endl;
    }

    cl::Platform::get( &OpenCLBase::all_platforms ) ;

    if( OpenCLBase::all_platforms.size() == 0  ) {
        std::string err_str = __func__;
        err_str += "\nNo OpenCL platforms found, check OpenCL installation";
        throw std::runtime_error( err_str );
    }

    OpenCLBase::default_platform = OpenCLBase::all_platforms[ platform_number ];
    OpenCLBase::default_platform.getDevices(CL_DEVICE_TYPE_ALL, &OpenCLBase::all_devices);

    if( device_number >= OpenCLBase::all_devices.size() ) {
        std::string err_str = __func__;
        err_str += "\nRequested device number of ";
        err_str += boost::lexical_cast<std::string>( device_number );
        err_str += " does not correspond to any device.\n";
        err_str +="Largest device number is ";
        err_str += boost::lexical_cast<std::string>( OpenCLBase::all_devices.size() );
        throw std::runtime_error( err_str );
    }

    OpenCLBase::current_device = OpenCLBase::all_devices[ device_number ];

    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(default_platform)(), 0};

//    OpenCLBase::context = cl::Context( clCreateContext( props, 1, &current_device(), NULL, NULL, &err_0 ); )
//    OpenCLBase::context = cl::Context ( {OpenCLBase::current_device} );
    OpenCLBase::context = cl::Context ( OpenCLBase::current_device, props );

    cl_int err = 0;
    OpenCLBase::command_queue = cl::CommandQueue (OpenCLBase::context,OpenCLBase::current_device, err);
    //OCL_DEBUG( err );

    if( DEBUG_ON ) {

        std::cout << "OpenCL Platform & Device info:" << std::endl;
        std::cout << "Platform Info: " <<default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "Device Info: " << current_device.getInfo<CL_DEVICE_NAME>() << std::endl;

    }

}

OpenCLBase::OpenCLBase(uint platform_number, uint device_number ) {

    //Static variables are initalized once and only once
    //these variables need to be consistent across all derived classes
    if( initalized != true ) {
        SetUp( platform_number, device_number );
    }

    initalized |= true;
}

OpenCLBase::~OpenCLBase() {
    if( DEBUG_ON )
        std::cout << __func__ << std::endl;
}

}
