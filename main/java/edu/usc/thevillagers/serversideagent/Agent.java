package edu.usc.thevillagers.serversideagent;

import com.mojang.authlib.GameProfile;

import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.server.management.PlayerInteractionManager;
import net.minecraft.world.WorldServer;
import net.minecraftforge.fml.common.FMLCommonHandler;

public class Agent extends EntityPlayerMP {
	
	public Agent(WorldServer world, GameProfile profile) {
		super(FMLCommonHandler.instance().getMinecraftServerInstance(), world, profile, new PlayerInteractionManager(world));
	}

	@Override
	public void onUpdate() {
		super.onUpdate();
		this.onLivingUpdate();
	}
	
	@Override
	public void travel(float strafe, float vertical, float forward) {
		super.travel(strafe, vertical, forward);
	}
	
	@Override
	public void onLivingUpdate() {
		super.onLivingUpdate();
		this.travel(0F, 0F, 0);
	}
	
	@Override
	public void knockBack(Entity entityIn, float strength, double xRatio, double zRatio) {
		super.knockBack(entityIn, strength, xRatio, zRatio);
	}
}
