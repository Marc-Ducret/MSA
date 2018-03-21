package edu.usc.thevillagers.serversideagent;

import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import net.minecraftforge.fml.common.event.FMLServerStartingEvent;

@Mod(modid = ServerSideAgentMod.MODID, name = ServerSideAgentMod.NAME, version = ServerSideAgentMod.VERSION)
public class ServerSideAgentMod {
    public static final String MODID = "serversideagent";
    public static final String NAME = "Server Side Agent";
    public static final String VERSION = "1.0";

    @EventHandler
    public void preInit(FMLPreInitializationEvent event) {
    }

    @EventHandler
    public void init(FMLInitializationEvent event) {
    }
    
    @EventHandler
    public void serverLoad(FMLServerStartingEvent event) {
    	event.registerServerCommand(new CommandSpawn());
    }
}
