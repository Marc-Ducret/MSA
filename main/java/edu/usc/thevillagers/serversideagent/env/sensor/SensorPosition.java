package edu.usc.thevillagers.serversideagent.env.sensor;

import java.util.function.Function;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import net.minecraft.util.math.Vec3d;

public class SensorPosition extends Sensor {

	private final float normX, normY, normZ;
	private final Function<Agent, Vec3d> pos;
	
	/**
	 * Normalisation parameters (should be the maximum value along component), set to 0 to not include component
	 * @param normX
	 * @param normY
	 * @param normZ
	 */
	public SensorPosition(float normX, float normY, float normZ, Function<Agent, Vec3d> pos) {
		super(nonNull(normX) + nonNull(normY) + nonNull(normZ));
		this.normX = normX;
		this.normY = normY;
		this.normZ = normZ;
		this.pos = pos;
	}
	
	@Override
	public void sense(Agent agent) {
		int offset = 0;
		Vec3d p = pos.apply(agent);
		if(normX != 0) values[offset++] = (float) p.x / normX;
		if(normY != 0) values[offset++] = (float) p.y / normY;
		if(normZ != 0) values[offset++] = (float) p.z / normZ;
	}
	
	private static int nonNull(float f) {
		return f != 0 ? 1 : 0;
	}
}
