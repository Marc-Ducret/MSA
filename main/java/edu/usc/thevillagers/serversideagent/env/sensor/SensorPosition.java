package edu.usc.thevillagers.serversideagent.env.sensor;

import java.util.function.Function;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import net.minecraft.util.math.Vec3d;

/**
 * A sensor that senses a position and normalises it w.r.t.
 * ({@link #normX}, {@link #normY}, {@link #normZ}). Axis where norm is 0 are ignored.
 */
public class SensorPosition extends Sensor {

	private final float normX, normY, normZ;
	private final Function<Actor, Vec3d> pos;
	
	/**
	 * Normalisation parameters (should be the maximum value along component), set to 0 to not include component
	 * @param normX
	 * @param normY
	 * @param normZ
	 */
	public SensorPosition(float normX, float normY, float normZ, Function<Actor, Vec3d> pos) {
		super(nonNull(normX) + nonNull(normY) + nonNull(normZ));
		this.normX = normX;
		this.normY = normY;
		this.normZ = normZ;
		this.pos = pos;
	}
	
	@Override
	public void sense(Actor actor) {
		int offset = 0;
		Vec3d p = pos.apply(actor);
		if(normX != 0) values[offset++] = (float) p.x / normX;
		if(normY != 0) values[offset++] = (float) p.y / normY;
		if(normZ != 0) values[offset++] = (float) p.z / normZ;
	}
	
	private static int nonNull(float f) {
		return f != 0 ? 1 : 0;
	}
}
