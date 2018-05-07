package edu.usc.thevillagers.serversideagent;

import java.io.IOException;

import edu.usc.thevillagers.serversideagent.command.CommandEnvironment;
import edu.usc.thevillagers.serversideagent.command.CommandFastTick;
import edu.usc.thevillagers.serversideagent.command.CommandRecord;
import edu.usc.thevillagers.serversideagent.command.CommandTPS;
import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import edu.usc.thevillagers.serversideagent.proxy.Proxy;
import edu.usc.thevillagers.serversideagent.request.RequestManager;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.SidedProxy;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import net.minecraftforge.fml.common.event.FMLServerStartingEvent;
import net.minecraftforge.fml.common.event.FMLServerStoppingEvent;

/**
 * Main class of the Mod that is loaded by Forge on startup.
 */
@Mod(modid = ServerSideAgentMod.MODID, name = ServerSideAgentMod.NAME, version = ServerSideAgentMod.VERSION, acceptableRemoteVersions = "*")
public class ServerSideAgentMod {
    public static final String MODID = "serversideagent";
    public static final String NAME = "Server Side Agent";
    public static final String VERSION = "1.0";
    
    private EnvironmentManager envManager;
    private RequestManager reqManager;
    
    @SidedProxy(clientSide = "edu.usc.thevillagers.serversideagent.proxy.ClientProxy", serverSide = "edu.usc.thevillagers.serversideagent.proxy.ServerProxy")
    public static Proxy proxy;

    @EventHandler
    public void preInit(FMLPreInitializationEvent event) {
    }

    @EventHandler
    public void init(FMLInitializationEvent event) {
    	MinecraftForge.EVENT_BUS.register(this);
    	MinecraftForge.EVENT_BUS.register(proxy);
    }
    
    @EventHandler
    public void serverLoad(FMLServerStartingEvent event) throws IOException {
    	envManager = new EnvironmentManager();
    	reqManager = new RequestManager(envManager);
    	reqManager.startRequestServer(1337);
    	
    	event.registerServerCommand(new CommandEnvironment(envManager));
    	event.registerServerCommand(new CommandFastTick());
    	event.registerServerCommand(new CommandTPS());
    	event.registerServerCommand(new CommandRecord());
    }
    
    @EventHandler
    public void serverClosing(FMLServerStoppingEvent event) throws IOException {
    	envManager.clearEnvs();
    	reqManager.stopRequestServer();
    }
}
