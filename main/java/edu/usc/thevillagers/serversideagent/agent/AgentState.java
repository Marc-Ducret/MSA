package edu.usc.thevillagers.serversideagent.agent;

import net.minecraft.block.state.BlockStateBase;
import net.minecraft.block.state.IBlockState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

public class AgentState {
	
	public static final int OBS_DIST = 1;
	public static final int OBS_SIZE = 2 * OBS_DIST + 1;
	
	public float forward, strafe, momentumYaw, momentumPitch;
	public boolean jump, crouch, attack, use;
	
	public IBlockState[] blocks;
	public Vec3d relativePos;
	public float yaw, pitch;
	
	public AgentState() {
		forward = strafe = momentumYaw = momentumPitch = 0;
		jump = crouch = attack = use = false;
		
		blocks = new BlockStateBase[OBS_SIZE * OBS_SIZE * OBS_SIZE];
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
		BlockPos pos = a.getPosition().add(-OBS_DIST, 0, -OBS_DIST);
		for(int z = 0; z < OBS_SIZE; z++) {
			for(int y = 0; y < OBS_SIZE; y++) {
				for(int x = 0; x < OBS_SIZE; x++) {
					blocks[x + y * OBS_SIZE + z * OBS_SIZE * OBS_SIZE] = world.getBlockState(pos.add(x, z, y));
				}
			}
		}
		relativePos = a.getPositionVector().subtract(new Vec3d(pos));
		yaw = a.rotationYaw;
		pitch = a.rotationPitch;
	}
	
	public IBlockState getBlockStateRelativeToAgent(int dx, int dy, int dz) {
		if(MathHelper.abs(dx) > OBS_DIST || MathHelper.abs(dy) > OBS_DIST ||  MathHelper.abs(dz) > OBS_DIST) 
			throw new IllegalArgumentException(dx+" "+dy+" "+dz+" is too far from agent (max dist: "+OBS_DIST);
		dx += OBS_DIST;
		dy += OBS_DIST;
		dz += OBS_DIST;
		return blocks[dx + dy * OBS_SIZE + dz * OBS_SIZE * OBS_SIZE];
	}
	
	public IBlockState getBlockStateRelativeToAgent(BlockPos pos) {
		return getBlockStateRelativeToAgent(pos.getX(), pos.getY(), pos.getZ());
	}
}
