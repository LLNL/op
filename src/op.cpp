#include "op.hpp"
#include <dlfcn.h>

namespace op {

  // /// Dynamically load an Optimizer
  
  // std::unique_ptr<op::Optimizer> PluginOptimizer(std::string optimizer_path)
  // {

  //   void* optimizer_plugin = dlopen(optimizer_path.c_str(), RTLD_LAZY);

  //   //    std::unique_ptr<op::Optimizer> (*load_optimizer)(op::CallbackFn, op::CallbackFn);

  //   auto load_optimizer = (std::unique_ptr<op::Optimizer> (*)(op::CallbackFn, op::CallbackFn)) dlsym( optimizer_plugin, "load_optimizer");
    
  //   return load_optimizer();
  // }  

  
}
