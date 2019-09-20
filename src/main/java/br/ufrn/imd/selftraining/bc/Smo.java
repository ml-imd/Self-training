package br.ufrn.imd.selftraining.bc;

import br.ufrn.imd.selftraining.enums.BaseClassifierType;
import br.ufrn.imd.selftraining.enums.ClassifierType;
import br.ufrn.imd.selftraining.utils.Encryptor;

public class Smo extends Classifier {

	private String sel;
	private double c;
	private int N;
	private double L;
	private double P;
	private boolean M;
	private double V;
	private double W;
	private String K;
	
	public Smo() {
		super();
		this.name = BaseClassifierType.SMO.getInfo();
		this.classifierType = ClassifierType.BASE_CLASSIFIER;
	}

	/**
	 * @param C   double - The complexity constant C. (default 1.0);
	 * @param N   int - To define 0=normalize, 1=standardize and 2=neither. (default
	 *            0);
	 * @param L   double - The tolerance parameter. (default 1.0e-3);
	 * @param P   double - The epsilon for round-off error. (default 1.0e-12);
	 * @param M   boolean - To fit calibration models to SVM outputs. (default
	 *            false);
	 * @param V   double - The number of folds for the internal cross-validation.
	 *            (default -1.0);
	 * @param W   double - The random number seed. (default 1);
	 * @param K   String - The Kernel to use. (default
	 *            weka.classifiers.functions.supportVector.PolyKernel)
	 * @param sel String - Full name of calibration model, followed by options. (default
	 *            weka.classifiers.functions.Logistic)
	 */

	public void setParameters(String sel, double c, int n, double l, double p, boolean m, double v, double w, String k) {
		this.sel = sel;
		this.c = c;
		this.N = n;
		this.L = l;
		this.P = p;
		this.M = m;
		this.V = v;
		this.W = w;
		this.K = k;
	}

	public String getSel() {
		return sel;
	}

	public void setSel(String sel) {
		this.sel = sel;
	}
	
	public double getC() {
		return c;
	}

	public int getN() {
		return N;
	}

	public void setN(int n) {
		N = n;
	}

	public double getL() {
		return L;
	}

	public void setL(double l) {
		L = l;
	}

	public double getP() {
		return P;
	}

	public void setP(double p) {
		P = p;
	}

	public boolean isM() {
		return M;
	}

	public void setM(boolean m) {
		M = m;
	}

	public double getV() {
		return V;
	}

	public void setV(double v) {
		V = v;
	}

	public double getW() {
		return W;
	}

	public void setW(double w) {
		W = w;
	}

	public String getK() {
		return K;
	}

	public void setK(String k) {
		K = k;
	}

	public void setC(double c) {
		this.c = c;
	}

	@Override
	public void setParametersByDefault() {
		this.sel = "Default";
	}

	@Override
	public void buildClassifierId() {
		String id = "-sel" + getSel() + "-C" + getC();
		this.classifierId = new String(Encryptor.encryptSh1(id));
	}

}
