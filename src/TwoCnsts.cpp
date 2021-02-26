#include <map>
#include <vector>
#include <algorithm>
#include <nlopt.hpp>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <mpi.h>

#include "gtest/gtest.h"
#include "op.hpp"
#include "op_config.hpp"
#include "nlopt_op.hpp"
// #include <math.h>
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

double start_x = 0.7;
double start_y = 0.7;
int number_of_constraints = 2;

double my_fun(double x, double y){
  double f = std::pow(1.0 - x,  2.0);
  f += 100.0*std::pow(y - (x*x), 2.0);
  //  std::cout << "X: " << x << " Y: " << y << " F: " << f << "\n";
  return f;
}

double my_c1(double x, double y){
  double c1 = std::pow(x-1.0, 3.0);
  c1 -= y;
  c1 += 1.0;
  return c1;
}

double my_c2(double x, double y){
  double c2 = x + y;
  c2 -= 2.0;
  return c2;
}

double dfdx(double x, double y){
  // double df = my_fun(x + step_size, y) - my_fun(x, y);
  // df /= step_size;
  double df = -2.0 + (2.0 * x) - (400 * x * (y - std::pow(x, 2)));
  // df += 400*std::pow(x, 3.0);
  return df;
}

double dfdy(double x, double y){
  // double df = my_fun(x, y + step_size) - my_fun(x, y);
  // df /= step_size;
  double df = 200*y;
  df -= 200*x*x;
  return df;
}

double dc1dx(double x, double ){
  // double df = my_c1(x + step_size, y) - my_c1(x, y);
  // df /= step_size;
  double df = std::pow(x - 1.0, 2.0);
  df *= 3;
  return df;
}

double dc1dy(double , double ){
  double df =-1.0;
  return df;
}

double dc2dx(double , double ){
  double df = 1.0;
  return df;
}

double dc2dy(double , double ){
  double df = 1.0;
  return df;
}

double myfunc(unsigned , const double *x, double *grad, void *)
{
    if (grad) {
        grad[0] = dfdx(x[0], x[1]);
        grad[1] = dfdy(x[0], x[1]);
    }
    return my_fun(x[0], x[1]);
}

typedef struct {
    double a, b;
} my_constraint_data;

double c1_nl(unsigned , const double *x, double *grad, void *)
{
    if (grad) {
        grad[0] = dc1dx(x[0], x[1]);
        grad[1] = dc1dy(x[0], x[1]);
    }
    return my_c1(x[0], x[1]);
}

double c2_nl(unsigned , const double *x, double *grad, void *)
{
    if (grad) {
        grad[0] = dc2dx(x[0], x[1]);
        grad[1] = dc2dy(x[0], x[1]);
    }
    return my_c2(x[0], x[1]);
}

// this does nothing, but shows how to pass a pointer through your functions!
my_constraint_data data[2] = { {2,0}, {-1,1} };

TEST(TwoCnsts, nlopt_serial)
{
    std::cout << "CJ: test functional evals at starting point\n";
    double f = my_fun(start_x, start_y);
    double dfdx_start = dfdx(start_x, start_y);
    double dfdy_start = dfdy(start_x, start_y);
    double c1 = my_c1(start_x, start_y);
    double c2 = my_c2(start_x, start_y);
    double dc1dx_start = dc1dx(start_x, start_y);
    double dc1dy_start = dc1dy(start_x, start_y);
    double dc2dx_start = dc2dx(start_x, start_y);
    double dc2dy_start = dc2dy(start_x, start_y);
    std::cout << "f(start_x, start_y): " << f << "\n";
    std::cout << "dfdx(start_x, start_y): " << dfdx_start << "\n";
    std::cout << "dfdy(start_x, start_y): " << dfdy_start << "\n";
    std::cout << "c1(start_x, start_y): " << c1 << "\n";
    std::cout << "c2(start_x, start_y): " << c2 << "\n";
    std::cout << "dc1dx(start_x, start_y): " << dc1dx_start << "\n";
    std::cout << "dc1dy(start_x, start_y): " << dc1dy_start << "\n";
    std::cout << "dc2dx(start_x, start_y): " << dc2dx_start << "\n";
    std::cout << "dc2dy(start_x, start_y): " << dc2dy_start << "\n";

    double f_star = my_fun(1.0, 1.0);
    double dfdx_star = dfdx(1.0, 1.0);
    double dfdy_star = dfdy(1.0, 1.0);
    double c1_star = my_c1(1.0, 1.0);
    double c2_star = my_c2(1.0, 1.0);
    double dc1dx_star= dc1dx(1.0, 1.0);
    double dc1dy_star = dc1dy(1.0, 1.0);
    double dc2dx_star = dc2dx(1.0, 1.0);
    double dc2dy_star = dc2dy(1.0, 1.0);
    std::cout << "\nCJ: solution evals \n";
    std::cout << "f(1, 1): " << f_star << "\n";
    std::cout << "dfdx(1, 1): " << dfdx_star << "\n";
    std::cout << "dfdy(1, 1): " << dfdy_star << "\n";
    std::cout << "c1(1, 1): " << c1_star << "\n";
    std::cout << "c2(1, 1): " << c2_star << "\n";
    std::cout << "dc1dx(1, 1): " << dc1dx_star << "\n";
    std::cout << "dc1dy(1, 1): " << dc1dy_star << "\n";
    std::cout << "dc2dx(1, 1): " << dc2dx_star << "\n";
    std::cout << "dc2dy(1, 1): " << dc2dy_star << "\n";

    // pick gcmma algo
    // https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa
    nlopt::opt opt(nlopt::LD_MMA, 2);  // 2 design variables

    // nlopt::opt opt(nlopt::LD_SLSQP, 2);  // slsqp instead :)

    std::vector<double> lb(2);
    lb[0] = -1.5; lb[1] = -0.5;
    opt.set_lower_bounds(lb);

    std::vector<double> ub(2);
    ub[0] = 1.5; ub[1] = 2.5;
    opt.set_upper_bounds(ub);

    opt.set_min_objective(myfunc, NULL);
    my_constraint_data data[2] = { {2,0}, {-1,1} };
    opt.add_inequality_constraint(c1_nl, &data[0], 1e-8); // 1e-8 is allowable constraint violation
    opt.add_inequality_constraint(c2_nl, &data[1], 1e-8);

    opt.set_xtol_rel(1e-6);  // various tolerance stuff ;)
    opt.set_maxeval(1000); // limit to 1000 function evals (i think)
    std::vector<double> x(2);
    x[0] = start_y; x[1] = start_y;
    double minf;

    try{
        nlopt::result result = opt.optimize(x, minf);
        std::cout << "found minimum at f(" << x[0] << "," << x[1] << ") = "
            << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    EXPECT_NEAR(0., minf, 1.e-9);
    
}

TEST(TwoCnsts, nlopt_op)
{
    std::cout << "CJ: test functional evals at starting point\n";
    double f = my_fun(start_x, start_y);
    double dfdx_start = dfdx(start_x, start_y);
    double dfdy_start = dfdy(start_x, start_y);
    double c1 = my_c1(start_x, start_y);
    double c2 = my_c2(start_x, start_y);
    double dc1dx_start = dc1dx(start_x, start_y);
    double dc1dy_start = dc1dy(start_x, start_y);
    double dc2dx_start = dc2dx(start_x, start_y);
    double dc2dy_start = dc2dy(start_x, start_y);
    std::cout << "f(start_x, start_y): " << f << "\n";
    std::cout << "dfdx(start_x, start_y): " << dfdx_start << "\n";
    std::cout << "dfdy(start_x, start_y): " << dfdy_start << "\n";
    std::cout << "c1(start_x, start_y): " << c1 << "\n";
    std::cout << "c2(start_x, start_y): " << c2 << "\n";
    std::cout << "dc1dx(start_x, start_y): " << dc1dx_start << "\n";
    std::cout << "dc1dy(start_x, start_y): " << dc1dy_start << "\n";
    std::cout << "dc2dx(start_x, start_y): " << dc2dx_start << "\n";
    std::cout << "dc2dy(start_x, start_y): " << dc2dy_start << "\n";

    double f_star = my_fun(1.0, 1.0);
    double dfdx_star = dfdx(1.0, 1.0);
    double dfdy_star = dfdy(1.0, 1.0);
    double c1_star = my_c1(1.0, 1.0);
    double c2_star = my_c2(1.0, 1.0);
    double dc1dx_star= dc1dx(1.0, 1.0);
    double dc1dy_star = dc1dy(1.0, 1.0);
    double dc2dx_star = dc2dx(1.0, 1.0);
    double dc2dy_star = dc2dy(1.0, 1.0);
    std::cout << "\nCJ: solution evals \n";
    std::cout << "f(1, 1): " << f_star << "\n";
    std::cout << "dfdx(1, 1): " << dfdx_star << "\n";
    std::cout << "dfdy(1, 1): " << dfdy_star << "\n";
    std::cout << "c1(1, 1): " << c1_star << "\n";
    std::cout << "c2(1, 1): " << c2_star << "\n";
    std::cout << "dc1dx(1, 1): " << dc1dx_star << "\n";
    std::cout << "dc1dy(1, 1): " << dc1dy_star << "\n";
    std::cout << "dc2dx(1, 1): " << dc2dx_star << "\n";
    std::cout << "dc2dy(1, 1): " << dc2dy_star << "\n";

    // pick gcmma algo
    // https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa

    // set bounds on variables
    std::vector<double> x(2);
    x[0] = start_y; x[1] = start_y;
    
    op::Vector<std::vector<double>> variables (x,
					       [](){return std::vector<double>{-1.5, -0.5};},
					       [](){return std::vector<double>{ 1.5,  2.5};});

    /*
    opt.set_xtol_rel(1e-6);  // various tolerance stuff ;)
    opt.set_maxeval(1000); // limit to 1000 function evals (i think)
    */

    auto nlopt_options = op::NLopt::Options
      {.Int = {
	  {"maxeval", 1000}
	},
       .Double = {
	  {"xtol_rel", 1.e-6}
	},
       .String = {{}}
      };
    auto opt = op::NLopt(variables, nlopt_options);

    std::vector<double> grad(2);

    opt.update = []() {
      std::cout << "Called Update" << std::endl;
    };
    
    // not sure why structured binding doesn't work...
    auto [obj_eval, obj_grad] = op::wrapNLoptFunc([&](unsigned n, const double* x , double * grad, void * data) {
	opt.UpdatedVariableCallback();
	return myfunc(n, x, grad, data);
      });
    op::Objective obj(obj_eval, obj_grad);

    auto [c1_nl_eval, c1_nl_grad] = op::wrapNLoptFunc(c1_nl);
    op::Objective constraint1(c1_nl_eval, c1_nl_grad);
    auto [c2_nl_eval, c2_nl_grad] = op::wrapNLoptFunc(c2_nl);
    op::Objective constraint2(c2_nl_eval, c2_nl_grad);

    // declare where we want to save our minimum value
    double minf;
    
    // method we'll call to go
    auto go = [&]()  {
      // set objective
      opt.setObjective(obj);
      opt.constraint_tol = 1.e-8;
      opt.addConstraint(constraint1);
      opt.constraint_tol = 1.e-8;
      opt.addConstraint(constraint2);
      opt.nlopt_->optimize(x, minf);
    };

    opt.go = go;

    try{
      opt.Go();
        std::cout << "found minimum at f(" << x[0] << "," << x[1] << ") = "
            << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    EXPECT_NEAR(0, minf, 1.e-9);
    
}

TEST(TwoCnsts, nlopt_op_plugin)
{
    std::cout << "CJ: test functional evals at starting point\n";
    double f = my_fun(start_x, start_y);
    double dfdx_start = dfdx(start_x, start_y);
    double dfdy_start = dfdy(start_x, start_y);
    double c1 = my_c1(start_x, start_y);
    double c2 = my_c2(start_x, start_y);
    double dc1dx_start = dc1dx(start_x, start_y);
    double dc1dy_start = dc1dy(start_x, start_y);
    double dc2dx_start = dc2dx(start_x, start_y);
    double dc2dy_start = dc2dy(start_x, start_y);
    std::cout << "f(start_x, start_y): " << f << "\n";
    std::cout << "dfdx(start_x, start_y): " << dfdx_start << "\n";
    std::cout << "dfdy(start_x, start_y): " << dfdy_start << "\n";
    std::cout << "c1(start_x, start_y): " << c1 << "\n";
    std::cout << "c2(start_x, start_y): " << c2 << "\n";
    std::cout << "dc1dx(start_x, start_y): " << dc1dx_start << "\n";
    std::cout << "dc1dy(start_x, start_y): " << dc1dy_start << "\n";
    std::cout << "dc2dx(start_x, start_y): " << dc2dx_start << "\n";
    std::cout << "dc2dy(start_x, start_y): " << dc2dy_start << "\n";

    double f_star = my_fun(1.0, 1.0);
    double dfdx_star = dfdx(1.0, 1.0);
    double dfdy_star = dfdy(1.0, 1.0);
    double c1_star = my_c1(1.0, 1.0);
    double c2_star = my_c2(1.0, 1.0);
    double dc1dx_star= dc1dx(1.0, 1.0);
    double dc1dy_star = dc1dy(1.0, 1.0);
    double dc2dx_star = dc2dx(1.0, 1.0);
    double dc2dy_star = dc2dy(1.0, 1.0);
    std::cout << "\nCJ: solution evals \n";
    std::cout << "f(1, 1): " << f_star << "\n";
    std::cout << "dfdx(1, 1): " << dfdx_star << "\n";
    std::cout << "dfdy(1, 1): " << dfdy_star << "\n";
    std::cout << "c1(1, 1): " << c1_star << "\n";
    std::cout << "c2(1, 1): " << c2_star << "\n";
    std::cout << "dc1dx(1, 1): " << dc1dx_star << "\n";
    std::cout << "dc1dy(1, 1): " << dc1dy_star << "\n";
    std::cout << "dc2dx(1, 1): " << dc2dx_star << "\n";
    std::cout << "dc2dy(1, 1): " << dc2dy_star << "\n";

    // pick gcmma algo
    // https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa

    // set bounds on variables
    std::vector<double> x(2);
    x[0] = start_y; x[1] = start_y;
    
    op::Vector<std::vector<double>> variables (x,
					       [](){return std::vector<double>{-1.5,-0.5};},
					       [](){return std::vector<double>{ 1.5, 2.5};});

    /*
    opt.set_xtol_rel(1e-6);  // various tolerance stuff ;)
    opt.set_maxeval(1000); // limit to 1000 function evals (i think)
    */

    auto nlopt_options = op::NLopt::Options
      {.Int = {
	  {"maxeval", 1000}
	},
       .Double = {
	  {"xtol_rel", 1.e-6}
	},
       .String = {{}}
      };

    auto opt = op::PluginOptimizer<op::NLopt>("./lib/libnlopt_so.so", variables, nlopt_options);

    std::vector<double> grad(2);

    opt->update = []() {
      std::cout << "Called Update" << std::endl;
    };
    
    // not sure why structured binding doesn't work...
    auto [obj_eval, obj_grad] = op::wrapNLoptFunc([&](unsigned n, const double* x , double * grad, void * data) {
	opt->UpdatedVariableCallback();
	return myfunc(n, x, grad, data);
      });
    op::Objective obj(obj_eval, obj_grad);

    auto [c1_nl_eval, c1_nl_grad] = op::wrapNLoptFunc(c1_nl);
    op::Objective constraint1(c1_nl_eval, c1_nl_grad);
    auto [c2_nl_eval, c2_nl_grad] = op::wrapNLoptFunc(c2_nl);
    op::Objective constraint2(c2_nl_eval, c2_nl_grad);
    
    // method we'll call to go
    opt->go = [&]()  {
      // set objective
      opt->setObjective(obj);
      opt->constraint_tol = 1.e-8;
      opt->addConstraint(constraint1);
      opt->constraint_tol = 1.e-8;
      opt->addConstraint(constraint2);
      opt->nlopt_->optimize(x, opt->final_obj);
    };

    try{
      opt->Go();
        std::cout << "found minimum at f(" << x[0] << "," << x[1] << ") = "
		  << std::setprecision(10) << opt->Solution() << std::endl;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

EXPECT_NEAR(0., opt->Solution(), 1.e-9);
    
}

#ifdef USE_LIDO
TEST(TwoCnsts, nlopt_op_bridge)
{
    std::cout << "CJ: test functional evals at starting point\n";
    double f = my_fun(start_x, start_y);
    double dfdx_start = dfdx(start_x, start_y);
    double dfdy_start = dfdy(start_x, start_y);
    double c1 = my_c1(start_x, start_y);
    double c2 = my_c2(start_x, start_y);
    double dc1dx_start = dc1dx(start_x, start_y);
    double dc1dy_start = dc1dy(start_x, start_y);
    double dc2dx_start = dc2dx(start_x, start_y);
    double dc2dy_start = dc2dy(start_x, start_y);
    std::cout << "f(start_x, start_y): " << f << "\n";
    std::cout << "dfdx(start_x, start_y): " << dfdx_start << "\n";
    std::cout << "dfdy(start_x, start_y): " << dfdy_start << "\n";
    std::cout << "c1(start_x, start_y): " << c1 << "\n";
    std::cout << "c2(start_x, start_y): " << c2 << "\n";
    std::cout << "dc1dx(start_x, start_y): " << dc1dx_start << "\n";
    std::cout << "dc1dy(start_x, start_y): " << dc1dy_start << "\n";
    std::cout << "dc2dx(start_x, start_y): " << dc2dx_start << "\n";
    std::cout << "dc2dy(start_x, start_y): " << dc2dy_start << "\n";

    double f_star = my_fun(1.0, 1.0);
    double dfdx_star = dfdx(1.0, 1.0);
    double dfdy_star = dfdy(1.0, 1.0);
    double c1_star = my_c1(1.0, 1.0);
    double c2_star = my_c2(1.0, 1.0);
    double dc1dx_star= dc1dx(1.0, 1.0);
    double dc1dy_star = dc1dy(1.0, 1.0);
    double dc2dx_star = dc2dx(1.0, 1.0);
    double dc2dy_star = dc2dy(1.0, 1.0);
    std::cout << "\nCJ: solution evals \n";
    std::cout << "f(1, 1): " << f_star << "\n";
    std::cout << "dfdx(1, 1): " << dfdx_star << "\n";
    std::cout << "dfdy(1, 1): " << dfdy_star << "\n";
    std::cout << "c1(1, 1): " << c1_star << "\n";
    std::cout << "c2(1, 1): " << c2_star << "\n";
    std::cout << "dc1dx(1, 1): " << dc1dx_star << "\n";
    std::cout << "dc1dy(1, 1): " << dc1dy_star << "\n";
    std::cout << "dc2dx(1, 1): " << dc2dx_star << "\n";
    std::cout << "dc2dy(1, 1): " << dc2dy_star << "\n";

    // pick gcmma algo
    // https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa

    // set bounds on variables
    std::vector<double> x(2);
    x[0] = start_y; x[1] = start_y;
    
    op::Vector<std::vector<double>> variables (x,
					       [](){return std::vector<double>{-1.5,-0.5};},
					       [](){return std::vector<double>{ 1.5, 2.5};});

    /*
      opt.set_xtol_rel(1e-6);  // various tolerance stuff ;)
      opt.set_maxeval(1000); // limit to 1000 function evals (i think)
    */

    struct Settings {
   
      double tol = 1.0e-6;;
      double dh = 1.0e-5;;
      int max_it = 500;
      bool test_deriv = false;
      double acceptable_tol = 5.0e-2;
      int acceptable_iter = 15;
      char * optimizer_solver = "ipopt";
      double movlim = 0.2;
      double fabstol = 1e-4;
      double freltol = 1e-4;
      int fwindow = 3;
      double fscale = 0.;
      bool restart = false;
      std::unordered_map<std::string, std::string> string_options;
      std::unordered_map<std::string, int> integer_options;
      std::unordered_map<std::string, double> numeric_options;
    };

    Settings settings;
    settings.string_options["derivative_test_print_all"] = "yes";
    
    auto opt = op::PluginOptimizer<op::Optimizer>("../../../lido-2.0/build/debug/lib/libLIDO_BRIDGE.so",
						  variables, MPI_COMM_WORLD, 0, settings);


    std::vector<double> grad(2);

    auto default_update = opt->update;
    
    auto new_update = [&]() {
      //      std::cout << x[0] << " " << x[1] << std::endl;
      default_update();
    };

    opt->update = new_update;
    
    // not sure why structured binding doesn't work...
    auto [obj_eval, obj_grad] = op::wrapNLoptFunc([&](unsigned n, const double* x , double * grad, void * data) {
    	opt->UpdatedVariableCallback();
    	return myfunc(n, x, grad, data);
      });
op::Objective obj(obj_eval, obj_grad);

double constraint_tol = 1.e-8;

auto [c1_nl_eval, c1_nl_grad] = op::wrapNLoptFunc([&](unsigned n, const double* x , double * grad, void * data) {
    	opt->UpdatedVariableCallback();
    	return c1_nl(n, x, grad, data);
      });
op::Objective constraint1(c1_nl_eval, c1_nl_grad, -1.e50, 0.);

auto [c2_nl_eval, c2_nl_grad] = op::wrapNLoptFunc([&](unsigned n, const double* x , double * grad, void * data) {
    	opt->UpdatedVariableCallback();
    	return c2_nl(n, x, grad, data);
      });
    op::Objective constraint2(c2_nl_eval, c2_nl_grad, -1.e50, 0.);

    auto default_go = opt->go;
    
    // method we'll call to go
    auto go = [&]()  {
      // set objective
      opt->setObjective(obj);
      opt->addConstraint(constraint1);
      opt->addConstraint(constraint2);

      // use our preset go function
      default_go();
    };

    opt->go = go;

    try{
      opt->Go();
        std::cout << "found minimum at f(" << x[0] << "," << x[1] << ") = "
		  << std::setprecision(10) << opt->Solution() << std::endl;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    EXPECT_NEAR(0., opt->Solution(), 1.e-9);
    
}
#endif
       

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  
  return RUN_ALL_TESTS();
}
