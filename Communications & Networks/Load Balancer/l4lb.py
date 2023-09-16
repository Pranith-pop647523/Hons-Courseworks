from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_4
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import in_proto
from ryu.lib.packet import arp
from ryu.lib.packet import ipv4
from ryu.lib.packet import tcp
from ryu.lib.packet.tcp import TCP_SYN
from ryu.lib.packet.tcp import TCP_FIN
from ryu.lib.packet.tcp import TCP_RST
from ryu.lib.packet.tcp import TCP_ACK
from ryu.lib.packet.ether_types import ETH_TYPE_IP, ETH_TYPE_ARP


class L4Lb(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(L4Lb, self).__init__(*args, **kwargs)
        self.ht = {}  # {(<sip><vip><sport><dport>): out_port, ...}
        self.vip = "10.0.0.10"
        self.dips = ("10.0.0.2", "10.0.0.3")
        self.dmacs = ("00:00:00:00:00:02", "00:00:00:00:00:03")
        #
        # write your code here, if needed
        self.dip_new = ""
        self.dmac_new = ""
        self.client_ip = "10.0.0.1"
        self.client_mac = "00:00:00:00:00:01"
        self.serverBit = 1  # 1 -> server 1 , 2 - > server 2
        #

    def _send_packet(self, datapath, port, pkt):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt.serialize()
        data = pkt.data
        actions = [parser.OFPActionOutput(port=port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=data,
        )
        return out

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def features_handler(self, ev):
        dp = ev.msg.datapath
        ofp, psr = (dp.ofproto, dp.ofproto_parser)
        acts = [psr.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.add_flow(dp, 0, psr.OFPMatch(), acts)

    def add_flow(self, dp, prio, match, acts, buffer_id=None):
        ofp, psr = (dp.ofproto, dp.ofproto_parser)
        bid = buffer_id if buffer_id is not None else ofp.OFP_NO_BUFFER
        ins = [psr.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, acts)]
        mod = psr.OFPFlowMod(
            datapath=dp, buffer_id=bid, priority=prio, match=match, instructions=ins
        )
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        in_port, pkt = (msg.match["in_port"], packet.Packet(msg.data))
        dp = msg.datapath
        ofp, psr, did = (dp.ofproto, dp.ofproto_parser, format(dp.id, "016d"))
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        #
        # write your code here, if needed
        #
        arph = pkt.get_protocols(arp.arp)
        tcph = pkt.get_protocols(tcp.tcp)
        iph = pkt.get_protocols(ipv4.ipv4)

        if arph:
            arpPacket = arph[0]
            opcode = arpPacket.opcode
            response = packet.Packet()
            if (
                opcode == 1
            ):  # we need to send respone if it is a request ( 1 == request )
                response_opcode = 2
                if eth.src == self.client_mac:
                    # req from client

                    ethNew = ethernet.ethernet(
                        self.client_mac, self.dmacs[0], ETH_TYPE_ARP
                    )

                    arpNew = arp.arp(
                        ETH_TYPE_ARP,
                        ETH_TYPE_IP,
                        6,
                        4,
                        response_opcode,
                        self.dmacs[0],
                        self.vip,
                        self.client_mac,
                        self.client_ip,
                    )
                    response.add_protocol(ethNew)  # eth has to be added before arp
                    response.add_protocol(arpNew)
                    out_port = 1

                elif eth.src == self.dmacs[0]:  # server 1
                    # req from server

                    ethNew = ethernet.ethernet(
                        self.dmacs[0], self.client_mac, ETH_TYPE_ARP
                    )

                    arpNew = arp.arp(
                        ETH_TYPE_ARP,
                        ETH_TYPE_IP,
                        6,
                        4,
                        response_opcode,
                        self.client_mac,
                        self.client_ip,
                        self.dmacs[0],
                        self.dips[0],
                    )
                    response.add_protocol(ethNew)
                    response.add_protocol(arpNew)

                elif eth.src == self.dmacs[1]:  # server 2
                    # req from server

                    ethNew = ethernet.ethernet(
                        self.dmacs[1], self.client_mac, ETH_TYPE_ARP
                    )

                    arpNew = arp.arp(
                        ETH_TYPE_ARP,
                        ETH_TYPE_IP,
                        6,
                        4,
                        response_opcode,
                        self.client_mac,
                        self.client_ip,
                        self.dmacs[1],
                        self.dips[1],
                    )
                    response.add_protocol(ethNew)
                    response.add_protocol(arpNew)
                dp.send_msg(self._send_packet(dp, in_port, response))  # sending reply

            return
        elif tcph and iph:
            match = None

            src_ip = iph[0].src
            dst_ip = iph[0].dst
            src_port = tcph[0].src_port
            dst_port = tcph[0].dst_port
            metadata_in = (src_ip, dst_ip, src_port, dst_port)

            if src_ip == self.client_ip and dst_ip == self.vip:

                if metadata_in in self.ht:
                    out_port = self.ht[metadata_in]
                else:
                    out_port = (
                        self.serverBit + 1
                    )  # using serverBit+1 as server 1 is in port 2 and server 2 is in port 3
                    self.ht[metadata_in] = out_port

                    if out_port == 2:
                        self.serverBit = 2
                    else:
                        self.serverBit = 1

                    # change next server to 2 if current port is server 1 and vice versa
                self.dmac_new = self.dmacs[
                    out_port % 2
                ]  # if port is 3 we want server 2 and 3 mod 2 gives 1 and returns second server details from list as it is 0 indexed
                self.dip_new = self.dips[out_port % 2]
                # changing dst mac and ip address
                acts = [
                    psr.OFPActionSetField(eth_dst=self.dmac_new),
                    psr.OFPActionSetField(ipv4_dst=self.dip_new),
                    psr.OFPActionOutput(out_port),
                ]
                match = psr.OFPMatch(
                    in_port=in_port,
                    eth_type=eth.ethertype,
                    ipv4_src=src_ip,
                    ipv4_dst=dst_ip,
                    ip_proto=iph[0].proto,
                    tcp_src=src_port,
                    tcp_dst=dst_port,
                )
            elif src_ip in self.dips and dst_ip == self.client_ip:
                # setting out port to client if src is either server
                out_port = 1
                # changing dst mac and ip addresss to server1 and vip
                acts = [
                    psr.OFPActionSetField(eth_src=self.dmacs[0]),
                    psr.OFPActionSetField(ipv4_src=self.vip),
                    psr.OFPActionOutput(out_port),
                ]
                match = psr.OFPMatch(
                    in_port=in_port,
                    eth_type=eth.ethertype,
                    ipv4_src=src_ip,
                    ipv4_dst=dst_ip,
                    ip_proto=iph[0].proto,
                    tcp_src=src_port,
                    tcp_dst=dst_port,
                )
        if match is not None:
            self.add_flow(dp, 1, match, acts, msg.buffer_id)
            if msg.buffer_id != ofp.OFP_NO_BUFFER:
                return
        else:
            return

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        out = psr.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=acts,
            data=data,
        )
        dp.send_msg(out)
