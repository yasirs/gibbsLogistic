#include <RcppArmadillo.h>
#include <vector>
#include <armadillo>
#include <cppbugs/cppbugs.hpp>
#include <algorithm>






using namespace cppbugs;
using std::cout;
using std::endl;

class TestModel: public MCModel {
public:
  const arma::ivec& y; // given
  const arma::mat& X; // given

  Normal<arma::vec> b;
  Deterministic<arma::vec> p_hat;
  Bernoulli<arma::ivec> likelihood;
  Deterministic<double> rsq;

  TestModel(const arma::ivec& y_,const arma::mat& X_, const arma::mat& b_init): y(y_), X(X_),
                                          b(b_init),
                                          p_hat(1/(1+arma::exp(-X*b.value))),
                                          likelihood(y_,true),
                                          rsq(0)
  {
    add(b);
    add(likelihood);
    add(rsq);
  }

  void update() {
    p_hat.value = 1/(1+exp(-X*b.value));
    rsq.value = arma::as_scalar(1 - var(y - p_hat.value) / var(y));
    b.dnorm(0.0, 0.0001);
    likelihood.dbern(p_hat.value);
  }
};

RcppExport SEXP logistic(SEXP x_, SEXP y_, SEXP iterations_, SEXP burnin_, SEXP adapt_, SEXP thin_) {
  arma::mat X = Rcpp::as<arma::mat>(x_);
  arma::ivec y = Rcpp::as<arma::ivec>(y_);
  const int iterations = Rcpp::as<int>(iterations_);
  const int burnin = Rcpp::as<int>(burnin_);
  const int adapt = Rcpp::as<int>(adapt_);
  const int thin = Rcpp::as<int>(thin_);

  const int NR = X.n_rows;
  const int NC = X.n_cols;

  TestModel m(y,X, arma::randn<arma::vec>(NC));
  m.p_hat.setSaveHistory(false);
  m.sample(iterations, burnin, adapt, thin);
  Rcpp::List ans;
  ans["b mean"] = m.b.mean();
  ans["mean likelihood"] = m.likelihood.meanLogLikelihood();
  return Rcpp::wrap(ans);
  /*
  typename std::list<double>::iterator max_logp_it = std::max_element(m.likelihood.logp_history.begin(), m.likelihood.logp_history.end());
  typename std::list<double>::iterator min_logp_it = std::min_element(m.likelihood.logp_history.begin(), m.likelihood.logp_history.end());
  typename std::list<vec>::iterator max_b_it = m.b.history.begin(); std::advance(max_b_it, std::distance(m.likelihood.logp_history.begin(),max_logp_it));
  typename std::list<vec>::iterator min_b_it = m.b.history.begin(); std::advance(min_b_it, std::distance(m.likelihood.logp_history.begin(),min_logp_it));
  cout << "data generated from b:" << endl << real_b;
  cout << "highest log likelihood attained in sampling:" << *max_logp_it << endl;
  cout << "highest likelihood sampled b:"<<endl<< *max_b_it;
  cout << "lowest log likelihood attained in sampling:" << *min_logp_it << endl;
  cout << "lowest likelihood sampled b:"<<endl<< *min_b_it;


  cout << "mean model likelihood:" << m.likelihood.meanLogLikelihood() << endl;
  cout << "b: " << endl << m.b.mean();
  cout << "R^2: " << m.rsq.mean() << endl;
  cout << "samples: " << m.b.history.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
  */
};







