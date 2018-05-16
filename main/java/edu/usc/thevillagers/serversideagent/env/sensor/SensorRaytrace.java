package edu.usc.thevillagers.serversideagent.env.sensor;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

public abstract class SensorRaytrace extends Sensor {
	
	public final int hRes, vRes;
	public final float fov, focal, ratio;
	public final float dist;

	public SensorRaytrace(int hRes, int vRes, float fov, float ratio) {
		super(hRes * vRes);
		this.hRes = hRes;
		this.vRes = vRes;
		this.fov = fov;
		this.focal = 1 / (2 * (float) Math.sin(Math.toRadians(fov / 2)));
		this.ratio = ratio;
		this.dist = focal * hRes;
	}

	@Override
	public void sense(Actor actor) {
		sense(actor.entity.world, actor.entity.getPositionEyes(1), actor.entity.rotationYaw, actor.entity.rotationPitch);
	}
	
	public void sense(World world, Vec3d from, float yaw, float pitch) {
		for(int v = 0; v < vRes; v++) {
			for(int h = 0; h < hRes; h++) {
				Vec3d dir = new Vec3d(
						(2 * h / (float) (hRes-1) - 1) * -1, 
						(2 * v / (float) (vRes-1) - 1) * -1 / ratio, 
						focal).normalize();
				dir = dir.rotatePitch((float) Math.toRadians(-pitch))
						 .rotateYaw((float) Math.toRadians(-yaw));
				values[v * hRes + h] = raytrace(world, from, dir);
			}
		}
	}
	
	private float raytrace(World world, Vec3d from, Vec3d dir) {
		RayTraceResult res = world.rayTraceBlocks(from, from.add(dir.scale(dist)));
		return encode(world, from, dir, res);
	}
	
	protected abstract float encode(World world, Vec3d from, Vec3d dir, RayTraceResult res);
}
