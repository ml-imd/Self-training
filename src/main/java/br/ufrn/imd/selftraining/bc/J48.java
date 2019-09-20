package br.ufrn.imd.selftraining.bc;

import br.ufrn.imd.selftraining.enums.BaseClassifierType;
import br.ufrn.imd.selftraining.enums.ClassifierType;
import br.ufrn.imd.selftraining.utils.Encryptor;

public class J48 extends Classifier {

	private boolean o;
	private boolean u;
	private boolean a;
	private boolean b;
	private boolean j;
	private boolean s;
	private int m;
	private double c;

	public J48() {
		super();
		this.name = BaseClassifierType.J48.getInfo();
		this.setClassifierType(ClassifierType.BASE_CLASSIFIER);
	}

	/**
	 * @param O boolean - To do not to collapse tree. (default false);
	 * @param U boolean - To use unpruned tree. (default false);
	 * @param A boolean - Laplace to smoothing for predicted probabilities. (default
	 *          false);
	 * @param B boolean - To use binary splites only. (default false);
	 * @param J boolean - To do not use MDL correction for info gain of numeric
	 *          attributes. (default false);
	 * @param S boolean - To don't perform subtree raising. (default false);
	 * @param M int - To set minimum number of instances per leaf. (default 2);
	 * @param C double - To set confidence threshold for pruning. (default 0.25);
	 */
	public void setParameters(boolean O, boolean U, boolean A, boolean B, boolean J, boolean S, int M, double C) {
		this.o = O;
		this.u = U;
		this.a = A;
		this.b = B;
		this.j = J;
		this.s = S;
		this.m = M;
		this.c = C;
	}

	@Override
	public void setParametersByDefault() {
		this.o = false;
		this.u = false;
		this.a = false;
		this.b = false;
		this.j = false;
		this.s = false;
		this.m = 2;
		this.c = .25;
	}

	@Override
	public void buildClassifierId() {
		String id = "-O" + getO() + "-U" + getU() + "-A" + getA() + "-B" + getB() + "-J" + getJ() + "-S"
				+ getS() + "-M" + getM() + "-C" + getC();
		this.classifierId = new String(Encryptor.encryptSh1(id));
	}

	public boolean getO() {
		return o;
	}

	public void setO(boolean o) {
		this.o = o;
	}

	public boolean getU() {
		return u;
	}

	public void setU(boolean u) {
		this.u = u;
	}

	public boolean getA() {
		return a;
	}

	public void setA(boolean a) {
		this.a = a;
	}

	public boolean getB() {
		return b;
	}

	public void setB(boolean b) {
		this.b = b;
	}

	public boolean getJ() {
		return j;
	}

	public void setJ(boolean j) {
		this.j = j;
	}

	public boolean getS() {
		return s;
	}

	public void setS(boolean s) {
		this.s = s;
	}

	public int getM() {
		return m;
	}

	public void setM(int m) {
		this.m = m;
	}

	public double getC() {
		return c;
	}

	public void setC(double c) {
		this.c = c;
	}

}
