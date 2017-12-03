#ifndef OPENCLBASE_H
#define OPENCLBASE_H

//C System-Headers
//
//C++ System headers
//
//OpenCL Headers
#include <CL/cl.h>
#include <CL/cl.hpp>
//Boost Headers
//
//OpenMP Headers
//
//Project specific headers
//

namespace ocl {

/*!
 * \brief Base class for every class that needs access to OpenCL
 * Platforms, Contexts or Devices.
 *
 * This class a common OpenCL Platform, Context, CommandQueue
 *  and Device for derived classes to share.
 */
class OpenCLBase {

  public:

    /*!
     * \brief Initialize OpenCL Objects
     * Note that the Platform and Context are selected automatically:
     * Platform is set to cl::Platform( 0 )
     * Context is set to cl::Context( device[ device_number ] )
     *
     * The underlying assumption is that the host machine has only
     * one OpenCL Platform ( e.g. one of AMD-APP, Nvidia CUDA, Intel )
     *
     * \param device_number
     *
     * Number of the OpenCL device to use
     * as described by cl::Platform::getDevices(CL_DEVICE_TYPE_ALL)
     */
    OpenCLBase( uint platform_number = 0, uint device_number = 0 );
    virtual ~OpenCLBase() = 0;

  protected:

    void SetUp( uint platform_number, uint device_number );

    static bool initalized;
    static std::vector<cl::Platform> all_platforms;
    static cl::Platform default_platform;
    static std::vector<cl::Device> all_devices;
    static cl::Device current_device;
    static cl::Context context;
    static cl::CommandQueue command_queue;
};

}

#endif // OPENCLBASE_H
