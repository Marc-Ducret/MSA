package edu.usc.thevillagers.serversideagent.env.sensor;

import edu.usc.thevillagers.serversideagent.ServerSideAgentMod;
import edu.usc.thevillagers.serversideagent.agent.Actor;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

/**
 * A sensor that uses ray tracing to produce some low resolution vision over multiple channels according to the specified encoding.
 */
public abstract class SensorRaytrace extends Sensor {
	
	public final int hRes, vRes, channels;
	public final float fov, focal, ratio;
	public float dist;
	public final boolean hitEntitites;
	
	public SensorRaytrace(int hRes, int vRes, int channels, float fov, float ratio) {
		this(hRes, vRes, channels, fov, ratio, false);
	}

	/**
	 * @param hRes: horizontal resolution
	 * @param vRes: vertical resolution
	 * @param channels: number of channels of the image
	 * @param fov: horizontal field of view (in degree)
	 * @param ratio: horizontal over vertical
	 */
	public SensorRaytrace(int hRes, int vRes, int channels, float fov, float ratio, boolean hitEntities) {
		super(hRes * vRes * channels);
		this.hRes = hRes;
		this.vRes = vRes;
		this.channels = channels;
		this.fov = fov;
		this.focal = 1 / (2 * (float) Math.sin(Math.toRadians(fov / 2)));
		this.ratio = ratio;
		this.dist = focal * hRes;
		this.hitEntitites = hitEntities;
	}

	@Override
	public void sense(Actor actor) {
		sense(actor.entity.world, actor.entity.getPositionEyes(1), actor.entity.rotationYaw, actor.entity.rotationPitch, actor.entity);
	}
	
	public void sense(World world, Vec3d from, float yaw, float pitch, Entity viewer) {
		float[] buffer = new float[channels];
		for(int v = 0; v < vRes; v++) {
			for(int h = 0; h < hRes; h++) {
				Vec3d dir = new Vec3d(
						(2 * h / (float) (hRes-1) - 1) * -1, 
						(2 * v / (float) (vRes-1) - 1) * -1 / ratio, 
						focal).normalize();
				dir = dir.rotatePitch((float) Math.toRadians(-pitch))
						 .rotateYaw((float) Math.toRadians(-yaw));
				raytrace(world, from, dir, viewer, buffer);
				for(int c = 0; c < channels; c++)
					values[(v * hRes + h) * channels + c] = buffer[c];
			}
		}
	}
	
	private void raytrace(World world, Vec3d from, Vec3d dir, Entity viewer, float[] result) {
		RayTraceResult res = ServerSideAgentMod.rayTrace(world, from, from.add(dir.scale(dist)), hitEntitites, viewer);
		preView(viewer);
		encode(world, from, dir, res, result);
	}
	
	protected void preView(Entity viewer) {
	}
	
	protected abstract void encode(World world, Vec3d from, Vec3d dir, RayTraceResult res, float[] result);
}
