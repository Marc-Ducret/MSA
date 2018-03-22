package edu.usc.thevillagers.serversideagent.agent;

import net.minecraft.block.state.BlockStateBase;
import net.minecraft.block.state.IBlockState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

public class AgentState {

	public final int updatePeriod;
	public final int obsDist;
	public final int obsSize;
	
	public float forward, strafe, momentumYaw, momentumPitch;
	public boolean jump, crouch, attack, use;
	
	public IBlockState[] blocks;
	public Vec3d relativePos;
	public float yaw, pitch;
	
	public AgentState(int updatePeriod, int obsDist) {
		this.updatePeriod = updatePeriod;
		this.obsDist = obsDist;
		this.obsSize = 2 * obsDist + 1;
		
		forward = strafe = momentumYaw = momentumPitch = 0;
		jump = crouch = attack = use = false;
		
		blocks = new BlockStateBase[obsSize * obsSize * obsSize];
		relativePos = Vec3d.ZERO;
		yaw = pitch = 0;
	}
	
	public void clamp() {
		forward = MathHelper.clamp(forward, -1, +1);
		strafe = MathHelper.clamp(strafe, -1, +1);
		momentumYaw = MathHelper.clamp(momentumYaw, -1, +1);
		momentumPitch = MathHelper.clamp(momentumPitch, -1, +1);
	}
	
	public void observe(Agent a) {
		World world = a.getEntityWorld();
		BlockPos pos = a.getPosition().add(-obsDist, 0, -obsDist);
		for(int z = 0; z < obsSize; z++) {
			for(int y = 0; y < obsSize; y++) {
				for(int x = 0; x < obsSize; x++) {
					blocks[x + y * obsSize + z * obsSize * obsSize] = world.getBlockState(pos.add(x, z, y));
				}
			}
		}
		relativePos = a.getPositionVector().subtract(new Vec3d(a.getPosition()));
		yaw = a.rotationYaw;
		pitch = a.rotationPitch;
	}
	
	public IBlockState getBlockStateRelativeToAgent(int dx, int dy, int dz) {
		if(MathHelper.abs(dx) > obsDist || MathHelper.abs(dy) > obsDist ||  MathHelper.abs(dz) > obsDist) 
			throw new IllegalArgumentException(dx+" "+dy+" "+dz+" is too far from agent (max dist: "+obsDist);
		dx += obsDist;
		dy += obsDist;
		dz += obsDist;
		return blocks[dx + dy * obsSize + dz * obsSize * obsSize];
	}
	
	public IBlockState getBlockStateRelativeToAgent(BlockPos pos) {
		return getBlockStateRelativeToAgent(pos.getX(), pos.getY(), pos.getZ());
	}
}
