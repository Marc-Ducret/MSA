package edu.usc.thevillagers.serversideagent.agent;

import java.util.ArrayList;
import java.util.List;

import net.minecraft.block.state.BlockStateBase;
import net.minecraft.block.state.IBlockState;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.util.math.Vec3i;
import net.minecraft.world.World;

public class AgentState { //TODO refactor to actor interface?

	public final int updatePeriod;
	public final int obsDist;
	public final int obsSize;
	
	public float forward, strafe, momentumYaw, momentumPitch;
	public boolean jump, crouch, attack, use;
	
	public Vec3d relativePos;
	public float yaw, pitch;
	public IBlockState[] blocks;
	public List<Entity> entities;
	
	public AgentState(int updatePeriod, int obsDist) {
		this.updatePeriod = updatePeriod;
		this.obsDist = obsDist;
		this.obsSize = 2 * obsDist + 1;
		
		forward = strafe = momentumYaw = momentumPitch = 0;
		jump = crouch = attack = use = false;
		
		relativePos = Vec3d.ZERO;
		yaw = pitch = 0;
		blocks = new BlockStateBase[obsSize * obsSize * obsSize];
		entities = new ArrayList<>();
	}
	
	public void clamp() {
		forward = MathHelper.clamp(forward, -1, +1);
		strafe = MathHelper.clamp(strafe, -1, +1);
		momentumYaw = MathHelper.clamp(momentumYaw, -1, +1);
		momentumPitch = MathHelper.clamp(momentumPitch, -1, +1);
	}
	
	public void observe(Agent a) {
		relativePos = a.getPositionVector();//.subtract(new Vec3d(a.getPosition())); TODO make relative?
		yaw = a.rotationYaw;
		pitch = a.rotationPitch;
		World world = a.getEntityWorld();
		Vec3i offset = new Vec3i(obsDist, obsDist, obsDist);
		BlockPos pos = a.getPosition().subtract(offset);
		for(int z = 0; z < obsSize; z++) {
			for(int y = 0; y < obsSize; y++) {
				for(int x = 0; x < obsSize; x++) {
					blocks[x + y * obsSize + z * obsSize * obsSize] = world.getBlockState(pos.add(x, y, z));
				}
			}
		}
		entities = a.world.getEntitiesWithinAABBExcludingEntity(a, new AxisAlignedBB(
				pos.subtract(offset), 
				pos.add(offset)));
	}
	
	public IBlockState getBlockStateRelativeToAgent(int dx, int dy, int dz) {
		if(MathHelper.abs(dx) > obsDist || MathHelper.abs(dy) > obsDist ||  MathHelper.abs(dz) > obsDist) 
			throw new IllegalArgumentException(dx+" "+dy+" "+dz+" is too far from agent (max dist: "+obsDist+")");
		dx += obsDist;
		dy += obsDist;
		dz += obsDist;
		return blocks[dx + dy * obsSize + dz * obsSize * obsSize];
	}
	
	public IBlockState getBlockStateRelativeToAgent(BlockPos pos) {
		return getBlockStateRelativeToAgent(pos.getX(), pos.getY(), pos.getZ());
	}
}
