package edu.usc.thevillagers.serversideagent.proxy;

import com.mojang.authlib.GameProfile;

import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.server.management.PlayerInteractionManager;
import net.minecraft.world.World;
import net.minecraft.world.WorldServer;
import net.minecraftforge.fml.common.FMLCommonHandler;

public class ServerProxy implements Proxy {

	@Override
	public EntityPlayer createReplayEntityPlayer(World world, GameProfile profile) {
		return new EntityPlayerMP(FMLCommonHandler.instance().getMinecraftServerInstance(), 
				(WorldServer) world, profile, new PlayerInteractionManager(world));
	}
}
