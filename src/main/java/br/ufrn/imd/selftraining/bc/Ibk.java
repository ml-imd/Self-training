package br.ufrn.imd.selftraining.bc;

import br.ufrn.imd.selftraining.enums.BaseClassifierType;
import br.ufrn.imd.selftraining.enums.ClassifierType; 
import br.ufrn.imd.selftraining.utils.Encryptor;

public class Ibk extends Classifier {

	private boolean e;
	private boolean i;
	private boolean f;
	private boolean x;
	private int k;
	private int w;
	private String a;

	public Ibk() {
		super();
		this.name = BaseClassifierType.IBK.getInfo();
		this.classifierType = ClassifierType.BASE_CLASSIFIER;
	}

	/**
	 * @param E boolean - To minimise mean squared error rather than mean absolute
	 *          error when using. (default false);
	 * @param I boolean - For to define weight neighbours by the inverse of their
	 *          distance. (default false);
	 * @param F boolean - For o define weight neighbours by 1 - their distance.
	 *          (default false);
	 * @param X boolean - To select the number of nearest neighbours between 1 and
	 *          the k value specified using hold-one-out evaluation on the training
	 *          data. (default false);
	 * @param K int - For to define number of nearest neighbours (k) used in
	 *          classification. (default 1);
	 * @param W int - Maximum number of training instances maintained. (default 0)
	 */
	public void setParameters(boolean E, boolean I, boolean F, boolean X, int K, int W, String A) {
		this.e = E;
		this.i = I;
		this.f = F;
		this.x = X;
		this.k = K;
		this.w = W;
		this.a = A;
	}

	public boolean getE() {
		return e;
	}

	public void setE(boolean e) {
		this.e = e;
	}

	public boolean getI() {
		return i;
	}

	public void setI(boolean i) {
		this.i = i;
	}

	public boolean getF() {
		return f;
	}

	public void setF(boolean f) {
		this.f = f;
	}

	public boolean getX() {
		return x;
	}

	public void setX(boolean x) {
		this.x = x;
	}

	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}
	
	public int getW() {
		return w;
	}

	public void setW(int w) {
		this.w = w;
	}
	
	public String getA() {
		return a;
	}

	public void setA(String a) {
		this.a = a;
	}

	@Override
	public void setParametersByDefault() {
		this.e = false;
		this.i = false;
		this.f = false;
		this.x = false;
		this.k = 1;
	}

	@Override
	public void buildClassifierId() {
		String id = "-E" + getE() + "-I" + getI() + "-F" + getF() + "-X" + getX() + "-K" + getK();
		this.classifierId = new String(Encryptor.encryptSh1(id));
	}


}
